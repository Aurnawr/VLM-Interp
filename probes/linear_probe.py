import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Harmful vs harmless linear probe on cached last-token hidden states (dataset2, prompts 0-399).
# Replaces the ad hoc "image harm accuracy" (mean-difference direction fit on even prompts,
# tested on odd ones, threshold never recorded) with:
#   - L2 logistic regression on z-scored features, per layer and modality
#   - 5-fold stratified CV; L2 strength picked by an inner 3-fold CV on each outer-train fold
#   - pooled out-of-fold AUROC and accuracy (threshold = the probe's own p=0.5, fit on train)
#   - stratified bootstrap 95% CIs over examples (conditional on the trained probes)
#   - label-shuffle null: the same pipeline (incl. inner CV) rerun on permuted labels
#   - reference rows: mean-difference direction with a midpoint threshold, both on the same
#     5 folds and on the original even/odd split
# Many problems (shuffles x L2 grid) share one X, so they are fit as one batch on the GPU.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAMES = {"llava": "LLaVA-1.5-7B", "qwen": "Qwen2.5-VL-7B"}
MODALITIES = ["text", "image"]
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]   # L2 on mean BCE (sklearn C = 1 / (lambda * n))
# Reference palette, categorical slots 1 and 2 (dataviz skill)
COLORS = {"text": "#2a78d6", "image": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def stratified_folds(y, k, rng):
    """Fold id per example, balanced per class."""
    fold = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        fold[idx] = np.arange(len(idx)) % k
    return fold


def auroc(scores, y):
    """scores [B, n], y [B, n] (0/1) -> [B]. Rank formula; float ties are negligible."""
    ranks = scores.argsort(-1).argsort(-1).double() + 1
    n1 = y.sum(-1).double()
    n0 = y.shape[-1] - n1
    return ((ranks * y).sum(-1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def zscore(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fit_logreg(X, Y, lam, iters):
    """L2 logistic regression for S label vectors at one lambda. X [n, D], Y [S, n] -> W [S, D], b [S].

    Problems are independent, so the summed loss is minimised jointly; L-BFGS is run per lambda so
    the batched problems have similar conditioning.
    """
    S, D = Y.shape[0], X.shape[1]
    W = torch.zeros(S, D, device=X.device, requires_grad=True)
    b = torch.zeros(S, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        z = W @ X.T + b[:, None]
        loss = F.binary_cross_entropy_with_logits(z, Y, reduction="none").mean(-1) + 0.5 * lam * (W ** 2).sum(-1)
        total = loss.sum()
        total.backward()
        return total

    opt.step(closure)
    return W.detach(), b.detach()


def decision(X, W, b):
    return W @ X.T + b[:, None]


def select_lambda(X, Y, inner_fold, iters):
    """Inner CV: mean AUROC per (label set, lambda) -> index of best lambda per label set [S]."""
    S = Y.shape[0]
    score = torch.zeros(S, len(LAMBDAS), device=X.device, dtype=torch.float64)
    for f in np.unique(inner_fold):
        tr, te = torch.as_tensor(inner_fold != f), torch.as_tensor(inner_fold == f)
        Xtr, Xte = zscore(X[tr], X[te])
        for j, lam in enumerate(LAMBDAS):
            W, b = fit_logreg(Xtr, Y[:, tr], lam, iters)
            score[:, j] += auroc(decision(Xte, W, b), Y[:, te])
    return score.argmax(-1)


def probe_cv(X, Y, fold, rng, iters):
    """Nested CV for S label sets at once. Returns out-of-fold logits [S, n] and chosen lambda idx [S, k]."""
    S, n = Y.shape
    oof = torch.zeros(S, n, device=X.device)
    chosen = np.zeros((S, fold.max() + 1), dtype=int)
    for f in range(fold.max() + 1):
        tr_np, te_np = fold != f, fold == f
        tr, te = torch.as_tensor(tr_np), torch.as_tensor(te_np)
        # Inner folds stratified on the true labels; shuffled label sets keep the same class counts
        inner = stratified_folds(Y[0, tr].cpu().numpy().astype(int), 3, rng)
        best = select_lambda(X[tr], Y[:, tr], inner, iters).cpu().numpy()
        chosen[:, f] = best
        Xtr, Xte = zscore(X[tr], X[te])
        for j in np.unique(best):
            rows = torch.as_tensor(np.where(best == j)[0])
            W, b = fit_logreg(Xtr, Y[rows][:, tr], LAMBDAS[j], iters)
            oof[rows[:, None], torch.where(te)[0][None, :]] = decision(Xte, W, b)
    return oof, chosen


def meandiff_scores(X, y, train, test):
    """Mean-difference direction fit on train; returns test projections minus the midpoint threshold."""
    mu1, mu0 = X[train & (y == 1)].mean(0), X[train & (y == 0)].mean(0)
    v = mu1 - mu0
    thr = 0.5 * (mu1 @ v + mu0 @ v)
    return X[test] @ v - thr


def bootstrap_ci(scores, y, n_boot, rng):
    """Stratified bootstrap over examples -> (AUROC lo, hi), (acc lo, hi)."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    idx = np.concatenate([rng.choice(pos, (n_boot, len(pos))), rng.choice(neg, (n_boot, len(neg)))], axis=1)
    s = torch.as_tensor(scores[idx])
    yy = torch.as_tensor(y[idx], dtype=torch.float64)
    a = auroc(s, yy).numpy()
    acc = ((s > 0).double() == yy).double().mean(-1).numpy()
    return np.percentile(a, [2.5, 97.5]), np.percentile(acc, [2.5, 97.5])


def run_layer(X, y, fold, n_shuffle, n_boot, rng, iters, device):
    n = len(y)
    Y = np.stack([y] + [rng.permutation(y) for _ in range(n_shuffle)]).astype(np.float32)
    Xg = torch.as_tensor(X, device=device)
    Yg = torch.as_tensor(Y, device=device)
    oof, chosen = probe_cv(Xg, Yg, fold, rng, iters)

    a = auroc(oof.double(), Yg.double()).cpu().numpy()
    acc = ((oof > 0).float() == Yg).float().mean(-1).cpu().numpy()
    s_true = oof[0].cpu().numpy().astype(np.float64)
    auc_ci, acc_ci = bootstrap_ci(s_true, y, n_boot, rng)

    # Mean-difference reference, same 5 folds and the original even/odd split
    Xd = Xg.double()
    yt = torch.as_tensor(y, device=device)
    md = torch.zeros(n, device=device, dtype=torch.float64)
    for f in range(fold.max() + 1):
        te = torch.as_tensor(fold == f, device=device)
        md[te] = meandiff_scores(Xd, yt, ~te, te)
    md_acc = ((md > 0).long() == yt).double().mean().item()
    md_auc = auroc(md[None], yt[None].double()).item()
    # Harmful rows first, then harmless; even/odd = position within each class (prompt index for dataset2)
    n1 = int(y.sum())
    even = torch.as_tensor(np.r_[np.arange(n1) % 2 == 0, np.arange(n - n1) % 2 == 0], device=device)
    eo = meandiff_scores(Xd, yt, even, ~even)
    eo_acc = ((eo > 0).long() == yt[~even]).double().mean().item()

    null_auc, null_acc = a[1:], acc[1:]
    return {
        "auroc": float(a[0]), "auroc_ci": auc_ci.tolist(),
        "acc": float(acc[0]), "acc_ci": acc_ci.tolist(),
        "null_auroc_mean": float(null_auc.mean()) if n_shuffle else None,
        "null_auroc_95": float(np.percentile(null_auc, 95)) if n_shuffle else None,
        "null_acc_95": float(np.percentile(null_acc, 95)) if n_shuffle else None,
        "p_shuffle": float((1 + (null_auc >= a[0]).sum()) / (1 + n_shuffle)) if n_shuffle else None,
        "lambda_chosen": [LAMBDAS[j] for j in chosen[0]],
        "meandiff5_acc": md_acc, "meandiff5_auroc": md_auc,
        "meandiff_evenodd_acc": eo_acc,
    }


def plot(results, title, path):
    mods = list(results)
    layers = sorted(int(l) for l in results[mods[0]])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (key, lo_key, ylabel) in zip(axes, [("auroc", "auroc_ci", "AUROC"), ("acc", "acc_ci", "Accuracy")]):
        for m in mods:
            r = [results[m][str(l)] for l in layers]
            ax.plot(layers, [x[key] for x in r], color=COLORS[m], lw=2, label=f"{m} probe")
            ax.fill_between(layers, [x[lo_key][0] for x in r], [x[lo_key][1] for x in r],
                            color=COLORS[m], alpha=0.2, lw=0)
        if key == "auroc":
            ax.plot(layers, [results["image"][str(l)]["meandiff_evenodd_acc"] for l in layers],
                    color=COLORS["image"], lw=1.5, ls=":", label="image, even/odd mean-diff (acc)")
        null_key = "null_auroc_95" if key == "auroc" else "null_acc_95"
        nulls = [max(results[m][str(l)][null_key] for m in mods) for l in layers
                 if results[mods[0]][str(l)][null_key] is not None]
        if nulls:
            ax.plot(layers, nulls, color=MUTED, lw=1.5, ls="--", label="shuffle null, 95th pct")
        ax.axhline(0.5, color=GRID, lw=1, zorder=0)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel, color=INK)
        ax.set_ylim(0.4, 1.01)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_table(results, header, args, path):
    lines = [header,
             f"probe = L2 logistic regression, z-scored features, {args.k}-fold stratified CV, lambda from inner 3-fold CV "
             f"over {LAMBDAS}",
             f"CI = stratified bootstrap ({args.n_boot}) over out-of-fold scores; null = {args.n_shuffle} label shuffles "
             f"through the same pipeline; p = shuffle p-value",
             "md5 = mean-difference direction + midpoint threshold on the same folds; "
             "eo = same on the original even/odd split (old Table 1 number)", ""]
    for m in results:
        lines.append(f"[{m}]")
        lines.append(f"{'layer':>5}  {'AUROC [95% CI]':>22}  {'acc [95% CI]':>22}  {'null95 AUROC':>12}  "
                     f"{'p':>6}  {'md5 acc':>7}  {'eo acc':>6}  lambda")
        for l in sorted(int(x) for x in results[m]):
            r = results[m][str(l)]
            null = f"{r['null_auroc_95']:.3f}" if r["null_auroc_95"] is not None else "-"
            p = f"{r['p_shuffle']:.3f}" if r["p_shuffle"] is not None else "-"
            lam = ",".join(f"{x:g}" for x in r["lambda_chosen"])
            lines.append(f"{l:>5}  {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f}, {r['auroc_ci'][1]:.3f}]  "
                         f"{r['acc']:.3f} [{r['acc_ci'][0]:.3f}, {r['acc_ci'][1]:.3f}]  {null:>12}  {p:>6}  "
                         f"{r['meandiff5_acc']:>7.3f}  {r['meandiff_evenodd_acc']:>6.3f}  {lam}")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), default="llava")
    parser.add_argument("--dataset", choices=["dataset2", "vlsu"], default="dataset2",
                        help="vlsu = unsafe vs safe natural images under one prompt (extract_vlsu.py), image only")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Default: every layer except 0 (constant template token)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n_shuffle", type=int, default=100)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=200, help="L-BFGS iterations per fit")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset == "vlsu":
        from vlsu_data import load_vlsu
        H1, H0, _ = load_vlsu(args.model)
        data = {"image": (H1, H0)}
        out_dir = os.path.join(ROOT, f"{args.model}-results", "vlsu_probe")
        header = (f"{MODEL_NAMES[args.model]}  VLSU natural images, one shared prompt: unsafe images (HS+HH, "
                  f"n={len(H1)}) vs safe images (SS+SH, n={len(H0)}), near-duplicates removed; last-token states")
    else:
        cache = np.load(os.path.join(ROOT, f"{args.model}-results", f"hidden_states_{args.model}.npz"))
        data = {m: (cache[f"harmful_{m}"], cache[f"harmless_{m}"]) for m in MODALITIES}
        out_dir = os.path.join(ROOT, f"{args.model}-results", "probe")
        n = len(data["text"][0])
        header = f"{MODEL_NAMES[args.model]}  dataset2 prompts 0-{n - 1} (n={n} per class)  last-token states"
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for m, (H1, H0) in data.items():
        y = np.r_[np.ones(len(H1)), np.zeros(len(H0))].astype(int)
        # Same folds for both modalities: row i is the same prompt in text and image
        fold = stratified_folds(y, args.k, np.random.default_rng(args.seed))
        layers = args.layers or list(range(1, H1.shape[1]))
        results[m] = {}
        for l in layers:
            X = np.concatenate([H1[:, l], H0[:, l]]).astype(np.float32)
            rng = np.random.default_rng([args.seed, l])
            r = run_layer(X, y, fold, args.n_shuffle, args.n_boot, rng, args.iters, device)
            results[m][str(l)] = r
            print(f"{m:5s} L{l:2d}  AUROC {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f},{r['auroc_ci'][1]:.3f}]  "
                  f"acc {r['acc']:.3f}  null95 {r['null_auroc_95']}  md5 {r['meandiff5_acc']:.3f}  "
                  f"eo {r['meandiff_evenodd_acc']:.3f}  lam {r['lambda_chosen']}", flush=True)

    with open(os.path.join(out_dir, "probe_results.json"), "w") as f:
        json.dump({"args": vars(args), "lambdas": LAMBDAS, "results": results}, f, indent=1)
    write_table(results, header, args, os.path.join(out_dir, "probe_table.txt"))
    data_name = "VLSU natural images" if args.dataset == "vlsu" else "dataset2"
    plot(results, f"{MODEL_NAMES[args.model]}, {data_name}: harmful vs harmless linear probe, last token "
                  f"(5-fold CV, 95% bootstrap CI)", os.path.join(out_dir, "probe_auroc.png"))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()

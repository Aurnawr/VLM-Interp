import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from linear_probe import ROOT, MODEL_NAMES, MODALITIES, stratified_folds, probe_cv

# Style controls for the harmful-vs-harmless probe (dataset2, prompts 0-399 per class).
#
# 1a. Surface baselines: classifiers that never see the VLM, fit on the same 5 outer folds as the
#     probe (C from an inner 3-fold CV). If one of them matches the probe, dataset2 is separable
#     without any understanding of harm, and the probe's AUROC says nothing beyond that.
# 1b. Matched-pair AUROC: AUROC is the share of (harmful, harmless) pairs ranked correctly. Here it
#     is computed only over pairs that share a style cue (same first word, same length, ...), using
#     the probe's out-of-fold scores. A probe that leans on a cue loses accuracy once the cue is
#     equal within every compared pair. Baselines are scored the same way.
#
# Probe out-of-fold scores are recomputed with linear_probe.probe_cv (same outer folds, no
# shuffles) and cached to <model>-results/probe/oof_scores.npz.

N = 400
CONDITIONS = {   # name -> description; pair masks built in pair_masks()
    "all": "all pairs (ordinary AUROC)",
    "no_question": "harmless prompts ending in '?' dropped (no harmful prompt is a question)",
    "first_word": "same first word",
    "length": "same word count",
    "lines": "same number of rendered lines in the image",
    "first_word_length": "same first word and word count within 2",
}
COND_COLORS = {"all": "#2a78d6", "no_question": "#eb6834", "first_word": "#1baf7a",
               "length": "#eda100", "lines": "#e87ba4", "first_word_length": "#4a3aa7"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_prompts():
    harmful = json.load(open(os.path.join(ROOT, "dataset2", "harmful.json")))[:N]
    harmless = json.load(open(os.path.join(ROOT, "dataset2", "harmless.json")))[:N]
    paths = [os.path.join(ROOT, "dataset2", "harmful_images", f"harmful_{i:03d}.png") for i in range(N)] + \
            [os.path.join(ROOT, "dataset2", "harmless", f"harmless_{i:03d}.png") for i in range(N)]
    return harmful + harmless, paths


def first_word(t):
    return t.split()[0].lower().strip(",.:;!?\"'“”")


def image_features(paths, thumb=(40, 30)):
    """Layout statistics and a downsampled grayscale thumbnail per rendered prompt."""
    layout, thumbs = [], []
    for p in paths:
        g = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        ink = g < 0.5
        rows = ink.any(1)
        # Text lines = runs of inked rows
        n_lines = int(np.sum(rows[1:] & ~rows[:-1]) + rows[0])
        ys, xs = np.where(ink)
        layout.append([ink.mean(), n_lines, ys.max() - ys.min() + 1, xs.max() - xs.min() + 1,
                       ink.any(0).sum()])
        thumbs.append(np.asarray(Image.open(p).convert("L").resize(thumb, Image.BILINEAR),
                                 dtype=np.float32).ravel() / 255.0)
    return np.array(layout), np.array(thumbs)


def simple_features(texts, n_lines):
    words = [t.split() for t in texts]
    return np.array([[len(w), len(t), np.mean([len(x) for x in w]), t.strip().endswith("?"),
                      t.count(","), any(c.isdigit() for c in t), any(c in t for c in "\"“”'"),
                      t.count(":"), l]
                     for t, w, l in zip(texts, words, n_lines)], dtype=np.float64)


def cv_scores(make_model, X, y, fold):
    """Out-of-fold decision scores on the probe's outer folds; C picked by inner 3-fold CV (AUROC)."""
    oof = np.zeros(len(y))
    for f in range(fold.max() + 1):
        tr, te = fold != f, fold == f
        Xtr = X[tr] if not isinstance(X, list) else [X[i] for i in np.where(tr)[0]]
        Xte = X[te] if not isinstance(X, list) else [X[i] for i in np.where(te)[0]]
        model, grid = make_model()
        gs = GridSearchCV(model, grid, scoring="roc_auc",
                          cv=StratifiedKFold(3, shuffle=True, random_state=0)).fit(Xtr, y[tr])
        oof[te] = gs.decision_function(Xte)
    return oof


def logreg(scale=True):
    def make():
        steps = ([StandardScaler()] if scale else []) + [LogisticRegression(max_iter=5000)]
        return make_pipeline(*steps), {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1, 10, 100]}
    return make


def surface_baselines(texts, paths, y, fold):
    layout, thumbs = image_features(paths)
    n_lines = layout[:, 1]
    fw = np.array([[first_word(t)] for t in texts])

    def onehot():
        return make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=5000)), \
               {"logisticregression__C": [1e-2, 1e-1, 1, 10, 100]}

    def tfidf():
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                             LogisticRegression(max_iter=5000)), \
               {"logisticregression__C": [1e-1, 1, 10, 100, 1000]}

    simple = simple_features(texts, n_lines)
    baselines = {
        "length": ("word + character count", cv_scores(logreg(), simple[:, :2], y, fold)),
        "first_word": ("first word only (one-hot)", cv_scores(onehot, fw, y, fold)),
        "format": ("length, mean word length, '?', commas, digits, quotes, colons, rendered lines",
                   cv_scores(logreg(), simple, y, fold)),
        "tfidf": ("bag of words + bigrams (TF-IDF): an ideal OCR reader with no understanding",
                  cv_scores(tfidf, texts, y, fold)),
        "layout": ("image layout: ink fraction, line count, text bounding box",
                   cv_scores(logreg(), layout, y, fold)),
        "pixels": ("40x30 grayscale thumbnail of the rendered image", cv_scores(logreg(), thumbs, y, fold)),
    }
    return baselines, n_lines


def pair_masks(texts, n_lines):
    """Boolean [N harmful, N harmless] masks of which pairs each condition compares."""
    fw = np.array([first_word(t) for t in texts])
    nw = np.array([len(t.split()) for t in texts])
    q = np.array([t.strip().endswith("?") for t in texts])
    h, s = slice(0, N), slice(N, 2 * N)
    same = lambda a: a[h][:, None] == a[s][None, :]
    return {
        "all": np.ones((N, N), bool),
        "no_question": np.broadcast_to(~q[s][None, :] & ~q[h][:, None], (N, N)).copy(),
        "first_word": same(fw),
        "length": same(nw),
        "lines": same(n_lines),
        "first_word_length": same(fw) & (np.abs(nw[h][:, None] - nw[s][None, :]) <= 2),
    }


def matched_auroc(scores, masks, n_boot, rng, device):
    """Pair-restricted AUROC with a stratified bootstrap CI over prompts.

    With C[i, j] = 1[s_harmful_i > s_harmless_j] (+0.5 for ties), AUROC over pairs in M is
    sum(C*M) / sum(M). A bootstrap resample is a multiplicity vector per class, so its AUROC is
    (w_h^T (C*M) w_s) / (w_h^T M w_s).
    """
    sh = torch.as_tensor(scores[:N], device=device, dtype=torch.float64)
    ss = torch.as_tensor(scores[N:], device=device, dtype=torch.float64)
    C = (sh[:, None] > ss[None, :]).double() + 0.5 * (sh[:, None] == ss[None, :]).double()
    wh = torch.as_tensor(rng.multinomial(N, np.full(N, 1 / N), n_boot), device=device, dtype=torch.float64)
    ws = torch.as_tensor(rng.multinomial(N, np.full(N, 1 / N), n_boot), device=device, dtype=torch.float64)
    out = {}
    for name, M in masks.items():
        M = torch.as_tensor(M, device=device, dtype=torch.float64)
        point = ((C * M).sum() / M.sum()).item()
        num = ((wh @ (C * M)) * ws).sum(-1)
        den = ((wh @ M) * ws).sum(-1)
        boot = (num / den).cpu().numpy()
        out[name] = {"auroc": point, "ci": np.percentile(boot[np.isfinite(boot)], [2.5, 97.5]).tolist()}
    return out


def probe_oof(model, y, fold, seed, iters, device, recompute):
    path = os.path.join(ROOT, f"{model}-results", "probe", "oof_scores.npz")
    if os.path.exists(path) and not recompute:
        print(f"Loading cached probe scores from {path}")
        c = np.load(path)
        return {m: c[m] for m in MODALITIES}
    cache = np.load(os.path.join(ROOT, f"{model}-results", f"hidden_states_{model}.npz"))
    out = {}
    for m in MODALITIES:
        H = np.concatenate([cache[f"harmful_{m}"], cache[f"harmless_{m}"]])
        L = H.shape[1]
        out[m] = np.zeros((L, len(y)), dtype=np.float32)
        Yg = torch.as_tensor(y[None].astype(np.float32), device=device)
        for l in range(1, L):
            Xg = torch.as_tensor(H[:, l].astype(np.float32), device=device)
            oof, _ = probe_cv(Xg, Yg, fold, np.random.default_rng([seed, l]), iters)
            out[m][l] = oof[0].cpu().numpy()
            print(f"probe {m:5s} L{l:2d} done", flush=True)
    np.savez(path, **out)
    return out


def plot(res, base, model, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, m in zip(axes, MODALITIES):
        layers = sorted(int(l) for l in res[m])
        for c in CONDITIONS:
            ax.plot(layers, [res[m][str(l)][c]["auroc"] for l in layers], color=COND_COLORS[c], lw=2,
                    label=CONDITIONS[c] if c != "all" else "all pairs")
        for b, ls in [("tfidf", "--"), ("format", ":")]:
            ax.axhline(base[b]["all"]["auroc"], color=MUTED, lw=1.5, ls=ls,
                       label=f"baseline: {'bag of words' if b == 'tfidf' else 'format features'} (all pairs)")
        ax.set_title(f"{m} input", color=INK, fontsize=11)
        ax.set_xlabel("Layer")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("AUROC over compared pairs", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"{MODEL_NAMES[model]}: probe AUROC when harmful/harmless pairs share a style cue", color=INK)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_table(res, base, masks, model, n_lines, path, show_layers):
    lines = [f"{MODEL_NAMES[model]}  style controls for the harmful-vs-harmless probe  "
             f"(dataset2 prompts 0-{N - 1}, same 5 outer folds as probe_table.txt)", "",
             "Matched-pair AUROC: AUROC over (harmful, harmless) pairs that share the cue only. 95% CI = stratified",
             "bootstrap (1000) over prompts. Baselines never see the VLM; they are the same for both models.", "",
             "Conditions (pairs compared / harmful prompts used / harmless prompts used):"]
    for c, d in CONDITIONS.items():
        M = masks[c]
        lines.append(f"  {c:18s} {d}  ({int(M.sum())} / {int(M.any(1).sum())} / {int(M.any(0).sum())})")
    lines += ["", "1a. Surface baselines (out-of-fold, same folds)"]
    lines.append(f"  {'baseline':12s}  " + "  ".join(f"{c:>20s}" for c in CONDITIONS))
    for b, r in base.items():
        lines.append(f"  {b:12s}  " + "  ".join(
            f"{r[c]['auroc']:.3f} [{r[c]['ci'][0]:.3f},{r[c]['ci'][1]:.3f}]" for c in CONDITIONS))
    for b, r in base.items():
        lines.append(f"    {b}: {r['desc']}")
    lines += ["", "1b. Probe, matched-pair AUROC [95% CI] at selected layers (every layer in controls_results.json)"]
    for m in MODALITIES:
        lines.append(f"[{m}]")
        lines.append(f"  {'layer':>5}  " + "  ".join(f"{c:>20s}" for c in CONDITIONS))
        for l in show_layers[m]:
            r = res[m][str(l)]
            lines.append(f"  {l:>5}  " + "  ".join(
                f"{r[c]['auroc']:.3f} [{r[c]['ci'][0]:.3f},{r[c]['ci'][1]:.3f}]" for c in CONDITIONS))
        worst = {c: min(res[m][l][c]["auroc"] for l in res[m]) for c in CONDITIONS}
        lines.append(f"  {'min':>5}  " + "  ".join(f"{worst[c]:>20.3f}" for c in CONDITIONS))
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), default="llava")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recompute", action="store_true", help="Refit the probe even if oof_scores.npz exists")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(ROOT, f"{args.model}-results", "probe")
    os.makedirs(out_dir, exist_ok=True)
    texts, paths = load_prompts()
    y = np.r_[np.ones(N), np.zeros(N)].astype(int)
    fold = stratified_folds(y, 5, np.random.default_rng(args.seed))   # same outer folds as linear_probe.py

    print("Fitting surface baselines...", flush=True)
    base_scores, n_lines = surface_baselines(texts, paths, y, fold)
    masks = pair_masks(texts, n_lines)
    rng = np.random.default_rng(args.seed)
    base = {}
    for b, (desc, s) in base_scores.items():
        base[b] = matched_auroc(s, masks, args.n_boot, rng, device)
        base[b]["desc"] = desc
        print(f"baseline {b:10s} " + "  ".join(f"{c} {base[b][c]['auroc']:.3f}" for c in CONDITIONS), flush=True)

    scores = probe_oof(args.model, y, fold, args.seed, args.iters, device, args.recompute)
    res = {}
    for m in MODALITIES:
        res[m] = {}
        for l in range(1, scores[m].shape[0]):
            res[m][str(l)] = matched_auroc(scores[m][l].astype(np.float64), masks, args.n_boot, rng, device)
        print(f"probe {m}: min over layers " + "  ".join(
            f"{c} {min(res[m][l][c]['auroc'] for l in res[m]):.3f}" for c in CONDITIONS), flush=True)

    L = scores["text"].shape[0] - 1
    show = sorted({1, 2, 4, L // 4, L // 2, 3 * L // 4, L})
    write_table(res, base, masks, args.model, n_lines, os.path.join(out_dir, "controls_table.txt"),
                {m: show for m in MODALITIES})
    with open(os.path.join(out_dir, "controls_results.json"), "w") as f:
        json.dump({"args": vars(args), "conditions": CONDITIONS,
                   "pairs": {c: int(masks[c].sum()) for c in CONDITIONS},
                   "baselines": base, "probe": res}, f, indent=1)
    plot(res, base, args.model, os.path.join(out_dir, "controls_matched.png"))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()

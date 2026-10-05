import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from linear_probe import ROOT, MODEL_NAMES, stratified_folds, auroc, bootstrap_ci
from controls import cv_scores, logreg
from vlsu_data import CLIP_PATH, NEAR_DUP_COS, keep_mask, load_vlsu

# VLSU natural images (unsafe HS+HH vs safe SS+SH, one shared prompt): the three numbers per model.
#   1. CLIP baseline: 5-fold CV logistic regression on CLIP ViT-L/14-336 features from LLaVA's own
#      frozen vision tower (CLS embedding, and the mean layer -2 patch feature LLaVA's projector reads).
#      What is decodable before the language model sees anything. Same images -> same for both models.
#   2. Final-token probe: read from probe_results.json written by
#      `linear_probe.py --dataset vlsu --n_shuffle 20`.
#   3. Projection onto the text refusal direction v_text (dataset2 harmful-text mean minus harmless-text
#      mean, identical to the saved refusal_text.pt): AUROC of h . v_text_hat, no training. Also the share
#      of images past the text threshold (midpoint of the harmful- and harmless-text means along v_text),
#      and the image class gap along v_text as a fraction of the text class gap.
# CIs: stratified bootstrap over images (CLIP and probe scores are out-of-fold).

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_PROBE, C_PROJ, C_CLIP, C_PATCH = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def projection_stats(H1, H0, v_text, t_mid, n_boot, rng, n_random=200):
    """Per-layer AUROC of the projection onto v_text_hat, with CI and text-threshold statistics."""
    y = np.r_[np.ones(len(H1)), np.zeros(len(H0))].astype(int)
    out = {}
    for l in range(1, H1.shape[1]):
        vhat = v_text[l] / np.linalg.norm(v_text[l])
        s = np.r_[H1[:, l] @ vhat, H0[:, l] @ vhat].astype(np.float64)
        a = auroc(torch.as_tensor(s)[None], torch.as_tensor(y, dtype=torch.float64)[None]).item()
        ci, _ = bootstrap_ci(s - np.median(s), y, n_boot, rng)
        text_gap = t_mid[l]["harmful"] - t_mid[l]["harmless"]
        # Random-direction null: AUROC of projections onto random unit vectors, two-sided (max(a, 1 - a))
        R = rng.standard_normal((n_random, H1.shape[2]))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        Sr = np.r_[H1[:, l], H0[:, l]] @ R.T
        ar = auroc(torch.as_tensor(Sr.T), torch.as_tensor(np.tile(y, (n_random, 1)), dtype=torch.float64)).numpy()
        out[str(l)] = {
            "auroc": a, "auroc_ci": ci.tolist(),
            "random_null_95": float(np.percentile(np.maximum(ar, 1 - ar), 95)),
            "cos_vtext_vimg": float(vhat @ (H1[:, l].mean(0) - H0[:, l].mean(0))
                                    / np.linalg.norm(H1[:, l].mean(0) - H0[:, l].mean(0))),
            "unsafe_past_text_threshold": float((s[y == 1] > t_mid[l]["thr"]).mean()),
            "safe_past_text_threshold": float((s[y == 0] > t_mid[l]["thr"]).mean()),
            "gap_frac_of_text_gap": float((s[y == 1].mean() - s[y == 0].mean()) / text_gap),
            "unsafe_mean_pos": float((s[y == 1].mean() - t_mid[l]["harmless"]) / text_gap),
            "safe_mean_pos": float((s[y == 0].mean() - t_mid[l]["harmless"]) / text_gap),
        }
    return out


def text_reference(model):
    """v_text per layer and the harmful/harmless text means along v_text_hat (dataset2, prompts 0-399)."""
    c = np.load(os.path.join(ROOT, f"{model}-results", f"hidden_states_{model}.npz"))
    Ht, H0t = c["harmful_text"].astype(np.float64), c["harmless_text"].astype(np.float64)
    v = Ht.mean(0) - H0t.mean(0)
    ref = {}
    for l in range(1, v.shape[0]):
        vhat = v[l] / np.linalg.norm(v[l])
        a, b = (Ht[:, l] @ vhat).mean(), (H0t[:, l] @ vhat).mean()
        ref[l] = {"harmful": a, "harmless": b, "thr": 0.5 * (a + b)}
    return v, ref


def clip_baseline(n_boot, rng):
    keep, dropped = keep_mask()
    c = np.load(CLIP_PATH)
    y_all = c["label"]
    idx = np.r_[np.where(keep & (y_all == 1))[0], np.where(keep & (y_all == 0))[0]]
    y = y_all[idx]
    fold = stratified_folds(y, 5, np.random.default_rng(0))   # same folds as linear_probe.py
    res = {}
    for name in ["clip_cls", "clip_patch"]:
        s = cv_scores(logreg(), c[name][idx].astype(np.float64), y, fold)
        a = auroc(torch.as_tensor(s)[None], torch.as_tensor(y, dtype=torch.float64)[None]).item()
        auc_ci, acc_ci = bootstrap_ci(s, y, n_boot, rng)
        res[name] = {"auroc": a, "auroc_ci": auc_ci.tolist(), "acc": float(((s > 0) == y).mean()),
                     "acc_ci": acc_ci.tolist()}
    return res, {"n_unsafe": int((y == 1).sum()), "n_safe": int((y == 0).sum()), "dropped": dropped}


def plot(probe, proj, clip, model, path):
    layers = sorted(int(l) for l in proj)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for res, key, color, label in [(probe, "image", C_PROBE, "final-token probe (5-fold CV)"),
                                   (proj, None, C_PROJ, "projection onto text refusal direction (no training)")]:
        r = [(res[key] if key else res)[str(l)] for l in layers]
        ax.plot(layers, [x["auroc"] for x in r], color=color, lw=2, label=label)
        ax.fill_between(layers, [x["auroc_ci"][0] for x in r], [x["auroc_ci"][1] for x in r],
                        color=color, alpha=0.2, lw=0)
    nulls = [probe["image"][str(l)]["null_auroc_95"] for l in layers]
    ax.plot(layers, nulls, color=MUTED, lw=1.5, ls=":", label="probe shuffle null, 95th pct")
    ax.plot(layers, [proj[str(l)]["random_null_95"] for l in layers], color=MUTED, lw=1.5, ls="-.",
            label="projection random-direction null, 95th pct")
    ax.axhline(clip["clip_cls"]["auroc"], color=C_CLIP, lw=1.5, ls="--", label="CLIP CLS embedding (vision encoder only)")
    ax.axhline(clip["clip_patch"]["auroc"], color=C_PATCH, lw=1.5, ls="--",
               label="CLIP layer -2 patch mean (LLaVA projector input)")
    ax.axhline(0.5, color=GRID, lw=1, zorder=0)
    ax.set_ylim(0.3, 1.01)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC, unsafe vs safe image", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"{MODEL_NAMES[model]}: VLSU natural images, one shared prompt", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), default="llava")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = os.path.join(ROOT, f"{args.model}-results", "vlsu_probe")
    rng = np.random.default_rng(args.seed)
    clip, info = clip_baseline(args.n_boot, rng)
    H1, H0, meta = load_vlsu(args.model)
    v_text, ref = text_reference(args.model)
    proj = projection_stats(H1.astype(np.float64), H0.astype(np.float64), v_text, ref, args.n_boot, rng)
    probe = json.load(open(os.path.join(out_dir, "probe_results.json")))["results"]

    layers = sorted(int(l) for l in proj)
    pk = max(layers, key=lambda l: probe["image"][str(l)]["auroc"])
    jk = max(layers, key=lambda l: proj[str(l)]["auroc"])
    L = layers[-1]
    f = lambda r: f"{r['auroc']:.3f} [{r['auroc_ci'][0]:.3f}, {r['auroc_ci'][1]:.3f}]"
    lines = [f"{MODEL_NAMES[args.model]}  VLSU natural images, one shared prompt",
             f"unsafe images (HS+HH) n={info['n_unsafe']}  vs  safe images (SS+SH) n={info['n_safe']}  "
             f"(dropped: {sum(d[0] == -1 for d in info['dropped'])} blank images, "
             f"{sum(d[0] != -1 for d in info['dropped'])} near-duplicates with CLIP CLS cosine > {NEAR_DUP_COS}; "
             f"no exact duplicates)",
             "AUROC [95% CI], stratified bootstrap over images", "",
             f"1. CLIP baseline (LLaVA's frozen ViT-L/14-336, 5-fold CV logistic regression)",
             f"   CLS embedding              {f(clip['clip_cls'])}   acc {clip['clip_cls']['acc']:.3f}",
             f"   layer -2 patch mean        {f(clip['clip_patch'])}   acc {clip['clip_patch']['acc']:.3f}",
             f"2. Final-token probe (linear_probe.py --dataset vlsu)",
             f"   peak (layer {pk:2d})            {f(probe['image'][str(pk)])}   acc {probe['image'][str(pk)]['acc']:.3f}",
             f"   last layer ({L:2d})            {f(probe['image'][str(L)])}   acc {probe['image'][str(L)]['acc']:.3f}",
             f"   shuffle null 95th pct      {max(probe['image'][str(l)]['null_auroc_95'] for l in layers):.3f} (max over layers)",
             f"3. Projection onto text refusal direction v_text (no training)",
             f"   best layer ({jk:2d})           {f(proj[str(jk)])}",
             f"   last layer ({L:2d})            {f(proj[str(L)])}",
             f"   random-direction null      {max(proj[str(l)]['random_null_95'] for l in layers):.3f} "
             f"(95th pct of max(AUROC, 1-AUROC) over 200 random unit directions; max over layers)", "",
             "Per layer: probe AUROC | projection AUROC | share of unsafe / safe images past the text threshold |",
             "image class gap along v_text as a fraction of the text class gap | mean unsafe / safe image position",
             "(0 = harmless-text mean, 1 = harmful-text mean) | projection random-direction null (95th pct) |",
             "cosine between v_text and the VLSU image class-mean difference (unsafe - safe images)",
             f"{'layer':>5}  {'probe':>22}  {'projection':>22}  {'unsafe>thr':>10}  {'safe>thr':>8}  {'gap frac':>8}  "
             f"{'unsafe pos':>10}  {'safe pos':>8}  {'rand null':>9}  {'cos(v_text, v_img)':>18}"]
    for l in layers:
        p, q = probe["image"][str(l)], proj[str(l)]
        lines.append(f"{l:>5}  {f(p):>22}  {f(q):>22}  {q['unsafe_past_text_threshold']:>10.3f}  "
                     f"{q['safe_past_text_threshold']:>8.3f}  {q['gap_frac_of_text_gap']:>8.3f}  "
                     f"{q['unsafe_mean_pos']:>10.2f}  {q['safe_mean_pos']:>8.2f}  {q['random_null_95']:>9.3f}  "
                     f"{q['cos_vtext_vimg']:>18.3f}")
    with open(os.path.join(out_dir, "vlsu_table.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "vlsu_results.json"), "w") as fh:
        json.dump({"clip": clip, "projection": proj, "probe_peak_layer": pk, "projection_best_layer": jk,
                   "n_unsafe": info["n_unsafe"], "n_safe": info["n_safe"], "near_duplicates_dropped": info["dropped"]},
                  fh, indent=1)
    plot(probe, proj, clip, args.model, os.path.join(out_dir, "vlsu_auroc.png"))
    print("\n".join(lines[:16]))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()

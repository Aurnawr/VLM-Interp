import os
import json
import argparse
import numpy as np
import torch

from linear_probe import ROOT, MODEL_NAMES, auroc

# In-format text refusal direction from VLSU itself, to replace dataset2's v_text in the VLSU analyses.
#   Safe images in both SH and SS, harmful vs safe VLSU text, the image-present chat template, so the direction
#   and its threshold live in the same input format as the VLSU image states, and the text harm covers
#   VLSU's categories instead of AdvBench's topics. From hidden_states_vlsu_own.npz (extract_vlsu.py --prompts own).
#   SH and SS use different images, and SH images are picked to go with harmful text (harm-adjacent though
#   graded safe), so the raw mean(SH) - mean(SS) also carries an image difference. The saved direction is
#   paired to cancel it: each item's own-prompt state minus the same image's shared-prompt state
#   (hidden_states_vlsu.npz), then mean over SH minus mean over SS:
#       v_text_vlsu = mean_SH(h_own - h_shared) - mean_SS(h_own - h_shared)
#   The raw mean(SH) - mean(SS) is kept as v_raw for reference.
# Saved: refusal_text_vlsu.pt = {"v": [layers, dim] float32 (paired), "v_raw", "harmful_mean"/"harmless_mean"/"thr":
#   [layers] (own-prompt SH / SS means along v_hat and their midpoint), "n_harmful", "n_harmless"}.
# Diagnostics per layer:
#   - norm of v_text_vlsu and of dataset2's v_text; share of the raw direction that is image difference
#     (|v_raw - v| / |v_raw|) and how well each separates the shared-prompt unsafe vs safe images
#     (the quantity the VLSU reruns measure, so image leakage here would inflate them)
#   - cos(v_text_dataset2, v_text_vlsu), with a random-direction null (95th pct |cos|)
#   - split-half reliability of v_text_vlsu (Spearman-Brown corrected, mean of 20 random splits)
#   - held-out checks (items not used to build the direction), AUROC of the projection, no training:
#       HH vs HS: unsafe images, harmful vs safe text, own prompts
#       dataset2 harmful vs harmless text (text only), projected onto v_text_vlsu
#     plus the same two checks for dataset2's v_text, for comparison (dataset2 text is in-sample there).


def unit(v):
    return v / np.linalg.norm(v)


def proj_auroc(X1, X0, vhat):
    s = np.r_[X1 @ vhat, X0 @ vhat].astype(np.float64)
    y = np.r_[np.ones(len(X1)), np.zeros(len(X0))]
    return auroc(torch.as_tensor(s)[None], torch.as_tensor(y, dtype=torch.float64)[None]).item()


def split_half(X1, X0, rng, n_splits=20):
    """Per-layer split-half cosine of the class-mean difference, Spearman-Brown corrected to full n."""
    r = np.zeros(X1.shape[1])
    for _ in range(n_splits):
        p1, p0 = rng.permutation(len(X1)), rng.permutation(len(X0))
        a1, b1 = p1[:len(p1) // 2], p1[len(p1) // 2:]
        a0, b0 = p0[:len(p0) // 2], p0[len(p0) // 2:]
        va = X1[a1].mean(0) - X0[a0].mean(0)
        vb = X1[b1].mean(0) - X0[b0].mean(0)
        c = (va * vb).sum(-1) / (np.linalg.norm(va, axis=-1) * np.linalg.norm(vb, axis=-1))
        r += 2 * c / (1 + c)
    return r / n_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), default="llava")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = os.path.join(ROOT, f"{args.model}-results", "vlsu_probe")
    rng = np.random.default_rng(args.seed)
    c = np.load(os.path.join(out_dir, "hidden_states_vlsu_own.npz"))
    H, cell = c["states"].astype(np.float64), c["cell"]
    sh = np.load(os.path.join(out_dir, "hidden_states_vlsu.npz"))
    assert (sh["uuid"] == c["uuid"]).all()
    D = H - sh["states"].astype(np.float64)   # own prompt minus shared prompt, same image
    SH, SS, HH, HS = (H[cell == k] for k in ["SH", "SS", "HH", "HS"])
    I1, I0 = (sh["states"][sh["label"] == k].astype(np.float64) for k in [1, 0])
    t = np.load(os.path.join(ROOT, f"{args.model}-results", f"hidden_states_{args.model}.npz"))
    Ht, H0t = t["harmful_text"].astype(np.float64), t["harmless_text"].astype(np.float64)

    v_raw = SH.mean(0) - SS.mean(0)
    v = D[cell == "SH"].mean(0) - D[cell == "SS"].mean(0)
    v_d2 = Ht.mean(0) - H0t.mean(0)
    rel = split_half(D[cell == "SH"], D[cell == "SS"], rng)
    layers = range(1, v.shape[0])
    hm, sm = np.zeros(v.shape[0]), np.zeros(v.shape[0])
    rows = {}
    for l in layers:
        vh, dh = unit(v[l]), unit(v_d2[l])
        hm[l], sm[l] = (SH[:, l] @ vh).mean(), (SS[:, l] @ vh).mean()
        R = rng.standard_normal((1000, v.shape[1]))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        rows[str(l)] = {
            "norm_vlsu": float(np.linalg.norm(v[l])), "norm_d2": float(np.linalg.norm(v_d2[l])),
            "cos_d2_vlsu": float(vh @ dh), "cos_null_95": float(np.percentile(np.abs(R @ vh), 95)),
            "split_half_sb": float(rel[l]),
            "image_share_raw": float(np.linalg.norm(v_raw[l] - v[l]) / np.linalg.norm(v_raw[l])),
            "shared_img_auroc": {"paired": proj_auroc(I1[:, l], I0[:, l], vh),
                                 "raw": proj_auroc(I1[:, l], I0[:, l], unit(v_raw[l])),
                                 "d2": proj_auroc(I1[:, l], I0[:, l], dh)},
            "vlsu_dir": {"HH_vs_HS": proj_auroc(HH[:, l], HS[:, l], vh),
                         "d2_text": proj_auroc(Ht[:, l], H0t[:, l], vh)},
            "d2_dir": {"HH_vs_HS": proj_auroc(HH[:, l], HS[:, l], dh),
                       "d2_text": proj_auroc(Ht[:, l], H0t[:, l], dh)},
        }

    torch.save({"v": torch.as_tensor(v, dtype=torch.float32), "v_raw": torch.as_tensor(v_raw, dtype=torch.float32),
                "harmful_mean": torch.as_tensor(hm),
                "harmless_mean": torch.as_tensor(sm), "thr": torch.as_tensor(0.5 * (hm + sm)),
                "n_harmful": len(SH), "n_harmless": len(SS)},
               os.path.join(out_dir, "refusal_text_vlsu.pt"))

    lines = [f"{MODEL_NAMES[args.model]}  in-format text direction v_text_vlsu = mean_SH(h_own - h_shared) - mean_SS(h_own - h_shared)",
             f"SH n={len(SH)}  SS n={len(SS)}  (safe images, harmful vs safe text); held out: HH n={len(HH)}, HS n={len(HS)}",
             "cos: cosine with dataset2 v_text (null = 95th pct |cos| with 1000 random unit vectors)",
             "split-half: reliability of v_text_vlsu, Spearman-Brown corrected, mean of 20 splits",
             "AUROC of the projection (no training): HH vs HS = harmful vs safe text on unsafe images (held out);",
             "d2 text = dataset2 harmful vs harmless text (held out for v_text_vlsu, in-sample for dataset2 v_text)",
             "img share = |v_raw - v| / |v_raw|, the image part of the unpaired mean(SH) - mean(SS)",
             "shared img = AUROC of shared-prompt unsafe vs safe images projected on: paired v / raw v / dataset2 v_text",
             f"{'layer':>5}  {'|v_vlsu|':>8}  {'|v_d2|':>8}  {'cos':>6}  {'null':>5}  {'split-half':>10}  "
             f"{'vlsu: HH/HS':>11}  {'vlsu: d2':>8}  {'d2: HH/HS':>9}  {'d2: d2':>6}  {'img share':>9}  "
             f"{'shared img p/raw/d2':>19}"]
    for l in layers:
        r = rows[str(l)]
        lines.append(f"{l:>5}  {r['norm_vlsu']:>8.2f}  {r['norm_d2']:>8.2f}  {r['cos_d2_vlsu']:>6.3f}  "
                     f"{r['cos_null_95']:>5.3f}  {r['split_half_sb']:>10.3f}  {r['vlsu_dir']['HH_vs_HS']:>11.3f}  "
                     f"{r['vlsu_dir']['d2_text']:>8.3f}  {r['d2_dir']['HH_vs_HS']:>9.3f}  {r['d2_dir']['d2_text']:>6.3f}  "
                     f"{r['image_share_raw']:>9.2f}  {r['shared_img_auroc']['paired']:>5.3f}/"
                     f"{r['shared_img_auroc']['raw']:.3f}/{r['shared_img_auroc']['d2']:.3f}")
    with open(os.path.join(out_dir, "vtext_vlsu_table.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "vtext_vlsu_results.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\n".join(lines))
    print(f"Saved refusal_text_vlsu.pt to {out_dir}")


if __name__ == "__main__":
    main()

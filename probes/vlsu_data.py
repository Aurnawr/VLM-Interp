import os
import numpy as np

# Shared loader for the VLSU natural-image set written by extract_vlsu.py, with duplicates removed.
# Exact duplicates (same file / same bytes) were checked when building the set: there are none.
# Near-duplicates are found from CLIP CLS embeddings (LLaVA's vision tower, saved by the LLaVA
# extraction): any pair with cosine > NEAR_DUP_COS is treated as the same image and only the first
# is kept. Blank images (pixel std < BLANK_STD; three all-black files in dataset_vlsu/, likely failed
# downloads) are dropped too. The kept set depends on the images only, so both models use the same rows.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_PATH = os.path.join(ROOT, "llava-results", "vlsu_probe", "clip_vlsu.npz")
NEAR_DUP_COS = 0.95
BLANK_STD = 3.0


def blank_rows(images):
    from PIL import Image
    return np.array([np.asarray(Image.open(os.path.join(ROOT, "dataset_vlsu", p)).convert("RGB"),
                                dtype=np.float32).std() < BLANK_STD for p in images])


def keep_mask():
    """Boolean mask over the 800 extracted rows; also returns the dropped rows as (kept_row or -1, dropped_row, cos)."""
    c = np.load(CLIP_PATH)
    E = c["clip_cls"].astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    keep = ~blank_rows(c["image"])
    dropped = [(-1, int(j), float("nan")) for j in np.where(~keep)[0]]
    for j in np.where(keep)[0]:
        for i in range(j):
            if keep[i] and S[i, j] > NEAR_DUP_COS:
                keep[j] = False
                dropped.append((int(i), int(j), float(S[i, j])))
                break
    return keep, dropped


def load_vlsu(model):
    """-> H1 (unsafe images), H0 (safe images) as [n, layers, dim], plus kept metadata (harmful rows first)."""
    c = np.load(os.path.join(ROOT, f"{model}-results", "vlsu_probe", "hidden_states_vlsu.npz"))
    keep, _ = keep_mask()
    y = c["label"]
    meta = {k: c[k] for k in ["cell", "uuid", "image", "image_category", "label"]}
    order = np.r_[np.where(keep & (y == 1))[0], np.where(keep & (y == 0))[0]]
    H = c["states"]
    return H[keep & (y == 1)], H[keep & (y == 0)], {k: v[order] for k, v in meta.items()}

import os
import re
import sys
import json
import math
import argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
from PIL import Image
from tqdm import tqdm

# Text-only control for the VLSU 2x2 behavioural baseline (old_project_scripts/.../vlsu_baseline.py).
# Qwen refuses only 0.25 of SH (safe image + unsafe text) vs 0.99 of AdvBench text. Either VLSU's
# unsafe texts are milder, or the image lowers refusal. Same VLSU prompts, re-run with:
#   --condition text  : no image at all, Exp-1 text format (USER: {text}\nASSISTANT: / chat template)
#   --condition blank : a uniform random-colour image (per item, seeded) in place of the real one, so
#                       "an image is present" is separated from "what the image shows"
# Everything else matches the baseline: prompts passed verbatim (wrapping quotes kept), greedy,
# 64 new tokens, same refusal scorer (Arditi substrings, imported from steer_cross_modal), and the
# outputs are paired per item with <model>-results/vlsu/baseline_generations.jsonl.
# 177/200 SH and 181/200 HH prompts point at the image ("this", "in the picture"), so without the
# image the model may answer "please provide the image", which the substring scorer can count as a
# refusal. Those are flagged (asks_for_image) and reported separately, along with a deictic split.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "old_project_scripts", "Experiment-2-scripts"))
from steer_cross_modal import MODELS, is_refusal, generate_batch  # noqa: E402

CELLS = ["SS", "HS", "SH", "HH"]
DEICTIC = re.compile(r"\b(this|these|that|those|here|shown|pictured|above|the (image|picture|photo))\b", re.I)
ASKS_FOR_IMAGE = re.compile(r"(provide|share|upload|see|view|access|without)\b[^.]{0,40}\b(image|picture|photo)"
                            r"|no (image|picture|photo)|(image|picture|photo)[^.]{0,30}\b(not|n't) (provided|attached|available)",
                            re.I)


def mcnemar_p(b, c):
    """Exact two-sided McNemar p from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), default="llava")
    parser.add_argument("--condition", choices=["text", "blank"], default="text")
    parser.add_argument("--data", default=os.path.join(ROOT, "dataset_vlsu"))
    parser.add_argument("--blank_side", type=int, default=448)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--gpu_mem", default="11GiB")
    args = parser.parse_args()

    cfg = MODELS[args.model]
    out_dir = os.path.join(ROOT, f"{args.model}-results", "vlsu")
    base_path = os.path.join(out_dir, "baseline_generations.jsonl")
    base = {(r["cell"], r["uuid"]): r for r in map(json.loads, open(base_path))}

    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
    model_cls = LlavaForConditionalGeneration if args.model == "llava" else Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(cfg["id"])
    processor.tokenizer.padding_side = "left"
    model = model_cls.from_pretrained(cfg["id"], dtype=cfg["dtype"], device_map="auto",
                                      max_memory={0: args.gpu_mem, "cpu": "96GiB"})
    model.eval()

    with_image = args.condition == "blank"

    def fmt(text):
        if args.model == "llava":
            return f"USER: <image>\n{text}\nASSISTANT:" if with_image else f"USER: {text}\nASSISTANT:"
        content = ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": text}]
        return processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)

    rng = np.random.default_rng(0)
    records = []
    for cell in CELLS:
        items = json.load(open(os.path.join(args.data, f"{cell}.json")))
        imgs = [Image.new("RGB", (args.blank_side,) * 2, tuple(int(v) for v in rng.integers(0, 256, 3)))
                for _ in items] if with_image else None
        outs = []
        for i in tqdm(range(0, len(items), args.batch_size), desc=cell):
            batch = items[i:i + args.batch_size]
            outs += generate_batch(model, processor, [fmt(it["prompt"]) for it in batch],
                                   imgs[i:i + args.batch_size] if with_image else None, args.max_new_tokens)
        for it, o in zip(items, outs):
            o = o.strip()
            b = base[(cell, it["uuid"])]
            records.append({"cell": cell, **it, "condition": args.condition, "completion": o,
                            "refusal": is_refusal(o), "asks_for_image": bool(ASKS_FOR_IMAGE.search(o)),
                            "deictic": bool(DEICTIC.search(it["prompt"])),
                            "image_completion": b["completion"], "image_refusal": b["refusal"]})
        print(f"  {cell}  refusal = {np.mean([r['refusal'] for r in records if r['cell'] == cell]):.3f}")

    tag = f"vlsu_{args.condition}_only" if args.condition == "text" else f"vlsu_{args.condition}_image"
    with open(os.path.join(out_dir, f"{tag}_generations.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cond = "no image" if args.condition == "text" else "random-colour image"
    lines = [f"{cfg['name']}  VLSU {cond} control  n=200 per cell  max_new_tokens={args.max_new_tokens}",
             f"same VLSU prompts; 'img' = original baseline (real image); refusal = substring scorer",
             "ctrl&!img = refuses here but not with the real image; p = exact McNemar (paired)",
             "clean = refusal excluding completions that ask for / say they can't see the image", "",
             f"{'cell':<6}{'ctrl':>7}{'img':>7}{'clean':>7}{'askimg':>8}{'ctrl&!img':>11}{'img&!ctrl':>11}{'p':>9}"
             f"   deictic (n) / non-deictic (n)"]
    for cell in CELLS:
        s = [r for r in records if r["cell"] == cell]
        ctrl = np.array([r["refusal"] for r in s])
        img = np.array([r["image_refusal"] for r in s])
        ask = np.array([r["asks_for_image"] for r in s])
        dei = np.array([r["deictic"] for r in s])
        b, c = int((ctrl & ~img).sum()), int((img & ~ctrl).sum())
        split = "  /  ".join(f"{ctrl[m].mean():.2f} ({m.sum()})" if m.any() else "- (0)" for m in [dei, ~dei])
        lines.append(f"{cell:<6}{ctrl.mean():>7.2f}{img.mean():>7.2f}{(ctrl & ~ask).mean():>7.2f}{ask.sum():>8}"
                     f"{b:>11}{c:>11}{mcnemar_p(b, c):>9.2g}   {split}")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(out_dir, f"{tag}_table.txt"), "w") as f:
        f.write(table + "\n")
    print(f"Done! Outputs in {out_dir}")


if __name__ == "__main__":
    main()

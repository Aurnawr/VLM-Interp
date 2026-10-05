# VLSU natural-image probe: predictions, written before extraction

Written 2026-10-05, before any VLSU hidden states, CLIP features or AUROCs were computed.
Inputs already seen: the dataset2 probe and control results, and the VLSU behavioural baselines
(`*/vlsu/baseline_table.txt`; e.g. Qwen refuses HS 0.01, HH 0.35; LLaVA HS 0.00, HH 0.13).

Setup: unsafe-image cells (HS + HH) vs safe-image cells (SS + SH), duplicates removed, every
image paired with the same prompt. Three numbers per model: CLIP baseline AUROC (5-fold CV logistic
regression on LLaVA's CLIP ViT-L/14-336 features), final-token probe AUROC (`linear_probe.py`,
every layer), and AUROC of the projection onto the saved text refusal direction `v_text` (no training).

## Claude's predictions

| Number | LLaVA-1.5-7B | Qwen2.5-VL-7B |
|---|---|---|
| CLIP baseline AUROC | 0.90–0.97 (unsafe VLSU images show weapons, blood, drugs: visually distinct) | same images, same value |
| Final-token probe, peak over layers | 0.88–0.97, about equal to CLIP | 0.90–0.98, about equal to or slightly above CLIP |
| Projection onto `v_text`, best layer | 0.55–0.70 | 0.60–0.80 |

Expected pattern: **probe high, projection near chance → "decodable but not routed"**, stronger
in LLaVA than in Qwen. Reasons: on typographic images LLaVA's `cos(v_text, v_image)` was only 0.45,
natural images should be further from text, and both models almost never refuse an unsafe image
with a safe prompt (HS refusal ≤ 0.01).

What would change my mind:
- A probe AUROC well below CLIP (e.g. < 0.75 while CLIP > 0.9) would point to information lost
  between the vision encoder and the final token: an encoding/transport failure rather than routing.
- A projection AUROC ≥ 0.85 with near-zero refusals would point to a gate threshold problem.

## User's predictions

(fill in before reading the results)

| Number | LLaVA-1.5-7B | Qwen2.5-VL-7B |
|---|---|---|
| CLIP baseline AUROC | | |
| Final-token probe, peak | | |
| Projection onto `v_text`, best layer | | |

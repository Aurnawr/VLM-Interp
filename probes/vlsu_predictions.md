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

## Outcome (added after the run, 2026-10-05; the sections above are unchanged)

| Number | LLaVA-1.5-7B | Qwen2.5-VL-7B | vs prediction |
|---|---|---|---|
| CLIP baseline AUROC (CLS) | 0.951 | 0.951 | inside 0.90–0.97 |
| Final-token probe, peak | 0.947 (L32) | 0.955 (L28) | inside range; ≈ CLIP as predicted |
| Projection onto `v_text`, best layer | 0.870 (L17) | 0.874 (L20) | above both predicted ranges |
| Random-direction null, same layer | 0.761 | 0.824 | not anticipated |

The projection AUROC miss is mostly a missing null: random directions alone reach 0.76–0.84,
so projection AUROC is a weak alignment measure here. On cosine and gap size, the pattern is "aligned
but too weak": cos(v_text, v_img) 0.35 (LLaVA) / 0.63 (Qwen); unsafe images move ≤ 8% / ≤ 16%
of the text harmful-harmless gap along v_text; 0% / ≤ 27% of unsafe images pass the text threshold.
"Would change my mind" conditions: the first (probe well below CLIP) did not occur. The second
(projection ≥ 0.85 with near-zero refusals → gate threshold problem) was met at face value
(0.87 in both models), but much of that AUROC is explained by the random-direction null. The cosine
and gap numbers point the same way, though: a partly aligned but small shift that stays below the
text threshold, which is a threshold/strength problem rather than a direction that is absent.

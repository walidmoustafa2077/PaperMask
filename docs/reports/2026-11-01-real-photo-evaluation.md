# Real-photo evaluation: v1 (legacy heatmap) vs v2 (Lite-FPN + offsets)

Date: 2026-11-01
Models: `outputs/heatmap_phase3/best.pth` (v1) and the Kaggle run's
`best.pth` (v2, epoch 107, val MAE 0.282 px on synthetic renders).
Data: `D:\Archive\Side Projects\Projects\REAL TEST` — 75 unlabeled real photos,
aspect ratios 1.0–2.29, 70 portrait / 5 landscape.

Scripts live in `%TEMP%\opencode\` and artefacts in `runs/real_test_compare/`
(gitignored). This file records what they showed.

## 1. The scores were never comparable

| | v1 3.79 px | v2 0.28 px |
|---|---|---|
| measured on | labels in `phase3_labels.json` | `nexadoc-50k/val` |
| who made the labels | v1 itself | Blender ground truth |
| domain | 1103 real photos | 1961 synthetic renders |

v1's number is resubstitution: it is scored against its own output, so it
measures self-consistency, not accuracy. v2's number is exact but in-domain.
Neither predicts behaviour on the other's data.

## 2. Same data, same preprocessing (220 repo photos)

Both models on the identical squashed 256×256 frame, error in 256-px units:

| | mean | median | p90 | images with a corner >20 px | IoU |
|---|---|---|---|---|---|
| v1 | 3.45 | 3.20 | 5.32 | ~1 % | 0.947 |
| v2 | 7.89 | 1.37 | 17.55 | 21.8 % | 0.913 |

v2's median is *better* than v1's but its tail is far worse. It is not broken;
it is confident only when the framing is familiar. On its weakest corner v2
averages 0.74 confidence when right and 0.25 when wrong — it knows.

Caveat: those labels are v1's output, so the comparison is biased toward v1 on
its own domain. The spread (median vs tail), not the mean, is the honest signal.

## 3. On the 75 unlabeled photos (squash, no gating)

| | v1 | v2 |
|---|---|---|
| forward, CPU, 256² | 29.1 ms | **18.5 ms** |
| edge support (quad on real gradients, >1 good) | 3.30 | **4.98** |
| min confidence, mean | **0.738** | 0.197 |
| valid quads | 75/75 | 72/75 |
| disagreement between them | 41.1 px mean / 83.6 p90 | |

v2 is 1.6× faster and sits on stronger edges, but its confidence collapses out
of domain and the two models disagree by 41 px on average — at least one is
badly wrong per image.

## 4. The real defect: framing, not accuracy

All training frames were square with the page at 60–70 % fill. Real photos are
1.0–2.29. Four projections of the same image move v2's prediction by
33–89 px — i.e. the model keys off the frame, not the page.

| aspect group | n | squash edge/conf | letterbox edge/conf | centre-crop edge/conf |
|---|---|---|---|---|
| ar ≤ 1.35 | 24 | 4.34 / 0.302 | 4.08 / 0.263 | 4.86 / 0.198 |
| 1.35 < ar ≤ 1.8 | 31 | 4.79 / 0.167 | 3.93 / 0.243 | 5.02 / 0.097 |
| ar > 1.8 | 20 | 6.04 / 0.117 | **7.76 / 0.744** | 4.79 / 0.075 |

Visual check (12 images across four aspect groups, `runs/real_test_compare/
montage/`): **letterbox (pad to square, grey fill) is correct almost every
time; squash is wrong almost every time on tall images.** Squash stretches a
2.29 frame into a square, which the model never saw, and it produces rotated,
skewed quads that miss the page entirely. Letterbox at ar 2.24 scores 0.86–0.96
confidence and traces the page border exactly.

On the 220 older *labeled* photos squash and letterbox are close (median 1.37
vs 1.61 px), with letterbox failing one case badly (`perspective_0053` 17.2 vs
0.67 px). Those photos are page-filling close-ups, not the deployment
distribution; the REAL TEST set is.

The centre-crop the .NET code currently uses is the worst of the three at tall
aspects — it throws away 25–50 % of the frame before the model sees it.

## 5. A second class of failure: no page boundary exists

Roughly 15 of the 75 images are scans, screenshots or photos where the document
fills the frame edge to edge. The corners are outside the image, so no model can
return them. v1 draws the image border, v2 draws an inner table or a rotated
quad. The product answer is not a better model — it is to detect
"quad ≈ frame" and pass the image through unwarped.

## 6. How to polish v2, in priority order

1. **Switch deployment preprocessing to letterbox.** No retraining. Fixes the
   tall-image class outright. Must be mirrored in `src/evaluate.py`,
   `refine_corners.py`, `src/export.py`'s parity fixture and (last, per the
   user's instruction) the C# decoder.
2. **Add framing augmentation to training** — random pad/crop to non-square
   aspect, random scale 0.5–1.2, random translate, then pad-to-square. This
   makes the projection choice stop mattering, which is the actual fix.
3. **Fine-tune on real photos.** Use v1 as a teacher (it is better in-domain),
   correct ~100–150 of the 220 labeled set with `refine_corners.py`, then
   fine-tune v2 at 1e-4 for 10–20 epochs with framing augmentation on. This
   moves the domain, not just the framing.
4. **Calibrate the confidence gate.** `model.json` ships 0.3, but v2's real-photo
   minimum score is 0.197 on average — the threshold would reject most correct
   predictions. Compute the operating point from a labeled real set before
   shipping.
5. **Handle the no-boundary case** explicitly: if the quad is within a few px of
   the frame, output the identity transform.
6. **Optional accuracy work**: higher-resolution offset head (stride 2 → 2×8×128²),
   multi-scale heatmap supervision, horizontal-flip TTA. Do these only after
   1–3, which address the actual failure.

## 7. Speed

v2 forward is 18.5 msCPU at 256², batch 1, ONNX-friendlier graph than v1
(1.6× faster). Preprocessing cost is dominated by the resize; letterbox adds one
`copyMakeBorder`. No inference-side work is needed.

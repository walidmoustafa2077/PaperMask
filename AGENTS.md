# PaperMask — Agent Handoff

Read this first. It is the compressed state of the repo so a new session does not have to re-explore.
Full findings: `AUDIT.md` (same directory).

## What this is

Document corner detection: predict 4 corner points of a photographed page, then perspective-warp it flat.

Two PyTorch models (MobileNetV3-Small backbone), both 256×256 input, 4 corners:

| Model | Neck | Output | Status |
|---|---|---|---|
| Point regression — `src/models/point_regression.py` | Sequential cross-attention over 5 scales | `corners` (B,8) sigmoid, normalized | **Untrainable as written** — attention at 1/2 scale OOMs (see below) |
| Heatmap regression — `src/models/heatmap_regression.py` | BiFPN, 4 scales (64/32/16/8) | `heatmaps` (B,4,64,64) sigmoid | Working; this is the one that shipped |

Deployment: `exports/heatmap_phase3.onnx` → `dotnet_project/Program.cs` (.NET 9 + OnnxRuntime 1.17 + OpenCvSharp 4.8) → argmax decode → warp → `outputs/dotnet_cropped/`.

## Map

```
src/
  train_point.py      point trainer            (CLI: --config --labels_path --images_dir --output_dir)
  train_heatmap.py    heatmap trainer          (same CLI; optional loss_type weighted_mse|focal|hybrid)
  export.py           checkpoint → TorchScript + ONNX (opset 14)
  data/dataset.py     CornerDataset, albumentations, Gaussian targets, 80/20 split
  losses/             smooth_l1.py, awing.py (unused), heatmap_focal.py
  metrics/geometry.py CornerMetrics: MAE/RMSE/PCK/IoU — computed in 256-space, ignores orig_size
  utils/postprocess.py heatmap_to_points (argmax|gaussian|centroid), scale/denormalize, convexity
  models/             point_regression.py, heatmap_regression.py
configs/              point_base.yaml, heatmap_base.yaml, heatmap_phase2.yaml, heatmap_phase3.yaml
image_labling.py            OpenCV click-labeler (4 corners, `+`/`-` zoom, no undo) — hardcoded dead paths
image_labling_validation.py renders every label onto its image — hardcoded dead paths
refine_corners.py           predict → drag corners → write JSON (also --auto, --export-onnx opset 11)
analyze_heatmap_model.py    post-hoc "learning vs memorization" report (imports scipy, hardcodes paths)
labels.json         169 labels  ↔ dataset_v2/   (0169.jpg …)
phase3_labels.json 1103 labels  ↔ dataset/      (curved_*, phone_* — the current set)
outputs/            heatmap_phase3/, heatmap_phase2_v2/, heatmap_regression*/ , point_regression/
exports/            heatmap_phase3.onnx (the deployable), exports.zip, Program.cs (duplicate of dotnet_project/Program.cs)
dotnet_project/     Program.cs + csproj + bin/ (742 MB, ignore)
torch_env/          4.9 GB broken venv — do not use
```

## Environment reality (do not assume)

- **`torch_env\Scripts\python.exe` is dead**: `pyvenv.cfg` points at `C:\Users\Walid\...\Python311`, which no longer exists. It is also the old path `D:\Side Projects\...`.
- No `.git` — the repo is **not version controlled**. No tests, no CI, no LICENSE file (README says MIT).
- 8.6 GB / 39.8k files. Datasets are 2.7 GB, `torch_env` 4.9 GB, `dotnet_project/bin` 742 MB.
- Python sources are few (~4k lines); the bulk is artifacts.

## The five things that will bite you

1. **Defaults are mismatched.** `labels.json` belongs to `dataset_v2/`, but the trainers default to `--images_dir dataset`. Mismatched pairs do not error — `CornerDataset` falls back to a black image on `imread` failure. Always pass the matching pair:
   `--labels_path phase3_labels.json --images_dir dataset`.
2. **Labels are model-generated pseudo-labels**, not human ground truth (all coordinates are non-integer on a 0.16-px grid; `image_labling.py` writes ints). Any MAE/IoU number is agreement with the previous model. A human-labeled held-out set does not exist yet.
3. **Metrics are in 256×256 space**, not original pixels (3024×4032). Multiply per-axis (11.8× / 15.75×) before comparing to deployment behavior.
4. **Point model OOMs**: at the final attention step the query is 128×128 = 16,384 tokens and K/V is the same → ~1.07e9 scores (~4.3 GB fp32) per sample with 4 heads. `point_base.yaml` asks for batch 4 on 4 GB. The shipped `point_regression/best.pth` therefore does not correspond to the current code path.
5. **`mobilenet_v3_large` is broken** in both models (hardcoded channel lists are Small's). Configs and QUICKSTART recommend it.

## Other confirmed defects (details + fixes in AUDIT.md)

- Export/refine ignore `fpn_channels` / `num_bifpn_blocks` (factory doesn't accept them); fine only while configs equal defaults 64/2.
- `refine_corners.py` reads `--config` YAML instead of `checkpoint['config']` → silent architecture drift.
- `with_classification: true` in all heatmap configs, but the head has no target and no loss → dead params; the shipped ONNX has no `cls_logits` (artifact ≠ repo code).
- `train_point.py` clips gradients **only on CPU**; AMP/CUDA path never clips. `train_heatmap.py` never clips despite `gradient_clip: 1.0` in phase3.
- Config keys that do nothing: `warmup_epochs`, `lr_schedule`, `gradient_clip`, `early_stopping_patience`, `save_best_only`, `log_interval`, `augment_*`.
- Augmentation can push corners outside the frame; they are `np.clip`ped to the border instead of dropped → trained-on-wrong-labels.
- Random split over burst filenames (`curved_0000…`) leaks near-duplicate frames into validation.
- Per-batch `cpu().numpy()` + `heatmap_to_points` + `metrics.compute()` in the tqdm loop → GPU starvation, O(n²) metric accumulation.
- `denormalize_corners` mutates its input tensor in place.
- `torch.load` without `weights_only=True` in export/refine/analyze → pickle RCE risk.
- C# deploy: argmax-only (no sub-pixel), assumes TL/TR/BR/BL, no convexity/clamping/degenerate check, non-recursive folder scan, basename collisions, `Mat`s not disposed, no stack traces.
- `QUICKSTART.md` references `test_installation.py`, `debug_visualize.py`, `plan-modelDesign.prompt.md` — none exist; claims 61 images (actual 1103).
- Annotation tool: saved on 4th click with no undo; pan offsets declared but never updated, so zoom is unusable; absolute `D:\Side Projects\...` paths are dead.

## Commands that work today

```powershell
# Only if a Python env exists elsewhere; .venv must be recreated first
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install scipy            # missing from requirements but imported

python src/train_heatmap.py --config configs/heatmap_phase3.yaml `
  --labels_path phase3_labels.json --images_dir dataset --output_dir outputs/heatmap_run

python refine_corners.py --checkpoint outputs/heatmap_phase3/best.pth `
  --config configs/heatmap_phase3.yaml --images-dir dataset --output phase3_refined.json --auto

python src/export.py --checkpoint outputs/heatmap_phase3/best.pth `
  --model_type heatmap --output_dir exports/heatmap_phase3_export --img_size 256 256

dotnet run --project dotnet_project      # interactive: file or folder path, q to quit
```

Note: `refine_corners.py` builds the model from `--config` (not the checkpoint) and accepts only `backbone` + `with_classification`, so keep `--config` equal to the training config.

## Data facts (verified)

- `labels.json` 169 ↔ `dataset_v2/` 169 — 0 missing/extra/malformed.
- `phase3_labels.json` 1103 ↔ `dataset/` 1103 — 0 missing/extra/malformed.
- All quads: 4 unique points, clockwise (TL,TR,BR,BL), non-self-intersecting.
- Images: 3024×4032 (dataset), 1536×2048 (dataset_v2); no EXIF surprises observed.
- Checkpoints: heatmap 12.99 MB, point 59.67 MB; each `outputs/<exp>/` also has `logs/` and `visualizations/epoch_XXX/` composites (GT | heatmaps | prediction | overlay).

## If you continue the work, start here

1. `git init` + `.gitignore` (`torch_env/`, `dataset*/`, `outputs/`, `bin/`, `obj/`, `__pycache__/`, `*.onnx`) + first commit — everything after is un-diffable otherwise.
2. Make the dataset fail loudly on missing images instead of substituting black.
3. Fix point-model attention (or drop the point model from the docs) and the large-backbone channel lists; add a one-batch forward test for both models/backbones.
4. Human-label a small blind test set; recompute MAE/IoU in original pixels against it.
5. Bring C# decoding to parity with Python (`centroid`/`gaussian`), add quad validation, then a 20-image ONNX↔PyTorch↔C# parity test.
6. Then: letterbox resize, group split, GPU-side metrics, config plumbing, CLI args for the tooling scripts.

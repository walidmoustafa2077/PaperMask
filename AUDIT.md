# Status: superseded (2026-10-30)

This audit describes the pre-rewrite codebase. The model, trainer, dataset
loader, loss, metrics, exporter and .NET app were all replaced on branch
`feat/litefpn-heatmap-offset` (spec:
`docs/superpowers/specs/2026-10-28-papermask-litefpn-retrain-design.md`).
Current architecture and commands: `AGENTS.md`, `README.md`.

Findings resolved by the rewrite:

| Finding | Disposition |
|---|---|
| P0-1 mismatched defaults / black-image fallback | Fixed. Single `data.root`, strict preflight, `FileNotFoundError` on unreadable images. |
| P0-2 dead committed venv | Fixed. `torch_env/` ignored; `.venv` documented. |
| P0-3 point-model attention OOM | Fixed by deletion. Point regression model retired. |
| P0-4 pseudo-labels invalidate metrics | Fixed by data source. Labels are exact geometry from the Blender generator. |
| P0-5 argmax-only C# decode | Fixed. Offset decode + quad validation; C# ↔ Python parity verified to 3 dp. |
| P1 metrics in 256-space | Intended. 256×256 is now the native resolution, so 256-space metrics are correct. |
| P1 large-backbone channel lists | Fixed. Channels are discovered at runtime. |
| P1 dead config knobs | Fixed. Strict schema; every key is consumed or rejected. |
| P1 clipping skipped under AMP | Fixed. `unscale_` → clip → step, both AMP and non-AMP paths. |
| P1 export/refine drop arch params | Fixed. Both rebuild through `build_model(checkpoint["config"])`. |
| P1 dead classification head / artifact drift | Fixed. Head removed; `model.json` describes the artifact. |
| P1 clamped augmented keypoints | Fixed. Augmentation rejects instead of clamping. |
| P1 square-resize distortion | Fixed. Generator emits square crops; C# uses a centred square crop. |
| P1 burst-split leakage | Fixed. Splits come from the generator and are never re-shuffled. |
| P1 dead absolute paths in labelers | Open. `image_labling*.py` kept as legacy; `refine_corners.py` is the supported tool. |
| P1 unsafe `torch.load` | Partially. Export/refine load local trusted checkpoints with `weights_only=False` on purpose; `arch_version` is validated. |
| P2 GPU starvation, O(n²) metrics, in-place denormalize, hook cache, no seeding | Fixed. GPU-side metrics, no hooks, `seed_everything`, no in-place tensor math. |
| P2 .NET resources/robustness | Fixed. Recursive scan, mirrored output paths, disposal, stack traces. |
| P2 doc drift, no tests, no LICENSE | Fixed. Docs rewritten; 30 tests; MIT LICENSE added. |

Still true: the first real training run has not happened yet, so no quality
number in this repository is trustworthy.

---

# PaperMask â€” Repository Audit

Date: 2026-10-28
Scope: whole repository (Python training pipeline, annotation/refinement tooling, ONNX export, .NET inference console app).
Method: static review of every first-party source file, data-integrity checks over both label sets, environment/artifact inspection. No training run or model execution was possible (see F-02).

---

## 1. Snapshot

| Item | Reality |
|---|---|
| Purpose | Document corner detection (4 corners) â€” point-regression model + Gaussian-heatmap model, exported to ONNX for a .NET/OpenCV console app |
| Version control | **None.** No `.git`, no `.gitignore`, no `LICENSE` file (README claims MIT) |
| Size | **8,657 MB / 39,775 files** |
| `torch_env/` | 4,896 MB / 37,758 files â€” a committed, **non-portable** venv |
| `dataset/` | 1,103 JPGs, 2,330 MB (curved_*/phone_* bursts, ~3024Ã—4032) |
| `dataset_v2/` | 169 JPGs, 390 MB (~1536Ã—2048), older set |
| `outputs/` | 296 MB of checkpoints + epoch visualizations, 5 experiment dirs |
| `dotnet_project/bin` | **742 MB** of checked-in build artifacts (3 runtime copies) |
| Tests / CI | **None** (no `tests/`, no pytest, no workflow) |
| Python env | Broken â€” `torch_env\Scripts\python.exe` exits with *â€œNo Python at 'C:\Users\Walid\...\Python311\python.exe'â€* |

Both label files are internally valid: 169/169 and 1103/1103 keys resolve to images, no malformed entries, no duplicate points, no self-intersecting quads, all clockwise. See F-04 for what that means.

---

## 2. Findings

Severity: **P0** = blocks correct use / destroys results; **P1** = wrong or misleading behavior; **P2** = quality, performance, hygiene.

### P0-1 â€” Default training command silently trains on black images
`src/train_heatmap.py` and `src/train_point.py` default to `--labels_path labels.json` and `--images_dir dataset`.
But `labels.json` (169 entries) belongs to `dataset_v2/`, and its keys (`0169.jpg`, â€¦) **do not exist in `dataset/`** (which holds `curved_0000.jpg`, `phone_*`, â€¦).
`CornerDataset.__getitem__` treats a failed `cv2.imread` as â€œuse a black 256Ã—256 imageâ€ and continues. Result: `python src/train_heatmap.py --config configs/heatmap_base.yaml` (the exact QUICKSTART command, and the README example) trains on 169 blank images with unrelated coordinates, forever, without failing.

**Fix:** validate at dataset construction that every label key resolves to a readable file; raise on the first missing key (or at least on >0% missing). Change defaults to the real pair, or drop defaults and make both args required. Never synthesize a blank sample.

### P0-2 â€” Committed virtualenv is broken and unreproducible
`torch_env/pyvenv.cfg`:
```
home = C:\Users\Walid\AppData\Local\Programs\Python\Python311
command = ... -m venv D:\Side Projects\Projects\PaperMask\torch_env
```
The repo now lives at `D:\Archive\...`, the interpreter user no longer exists, and the venv is 4.9 GB / 37.7k files. Every documented command (`.\torch_env\Scripts\activate`) fails on this machine.

**Fix:** delete `torch_env/` from the tree, rebuild a venv locally, pin with `pip freeze` if needed, add `.gitignore`.

### P0-3 â€” Point model cannot train at the documented configuration (attention memory blow-up)
`src/models/point_regression.py` â†’ `CrossAttentionNeck.forward` up-samples the query to the *next* scale **before** each attention step, and the 5th scale is `feat_1` at 1/2 resolution (128Ã—128 = 16,384 tokens at 256 input). At `i=4` the attention is `16384 Ã— 16384` per head; with `num_heads=4` that is ~1.07e9 scores â‰ˆ **4.3 GB fp32 for a single sample**, before K/V/activations. `configs/point_base.yaml` requests `batch_size: 4` on a 4 GB GPU.

The model is â€œdocumented and exportedâ€ but is effectively untrainable as written. `outputs/point_regression/best.pth` exists (59.7 MB), so either it was trained at a smaller size / with different code, or with heavy swapping.

**Fix:** attend only over 1/32â€¦1/8 scales, pool K/V to a fixed token budget, or use linear/deformable attention. Add a one-batch smoke test (`batch=2, 256Ã—256`) to CI.

### P0-4 â€” â€œGround truthâ€ labels are machine-generated, so reported accuracy is not accuracy
`image_labling.py` stores **integer** pixels (`int((x+offset)/scale)`). Both label files contain only non-integer coordinates on a uniform 0.16-px grid (sample from `labels.json`: `514.0800170898438, 1172.6400146484375, 2385.60009765625, â€¦`). They are perfectly canonical/clockwise/non-self-intersecting. That is the signature of model predictions (resized/re-scaled floats), not human clicks.

So `labels.json`, `phase3_labels.json`, and therefore every MAE/RMSE/PCK/IoU in `outputs/*` measure **agreement with the previous model**, not with independent ground truth. The documented â€œ5.79 px MAEâ€ and the claimed refinement improvement (QUICKSTART_REFINEMENT Â§Step 4) cannot be trusted as generalization numbers.

**Fix:** produce a human-labeled, model-blind test set (hundreds of images, corners clicked at native resolution) and report against that. Keep training labels, but label them as pseudo-labels in the README. Record how each file was produced.

### P0-5 â€” Deployment decoding is far worse than what the metrics measure
`dotnet_project/Program.cs` â†’ `PredictCorners` uses plain argmax on a 64Ã—64 heatmap, then `x = maxX * 4 * (W/256)`. Quantization is Â±4 px in 256-space, magnified by `W/256` â†’ up to **~63 px** of systematic error on a 4032-px photo. Python has centroid/gaussian refinement; the shipping path does not.

Also missing: channel-order validation (assumes TL,TR,BR,BL), convexity/self-intersection check, clamping to image bounds, and degenerate-quad handling (`width/height` can be 0 â†’ `WarpPerspective` empties/throws).

**Fix:** implement the same `centroid`/`gaussian` postprocess in C# (or export a model that outputs refined coordinates), validate the quad, clamp, and fail per-image instead of whole-folder.

### P1-1 â€” Metrics are computed in 256Ã—256 space only
`src/metrics/geometry.py` accepts `orig_size` but **never uses it**; it multiplies normalized coords by `self.img_size` (default 256) and normalizes by `sqrt(256Â²+256Â²)`. On 3024Ã—4032 inputs the real per-axis error is 11.8Ã—/15.75Ã— larger. `analyze_heatmap_model.py` repeats the assumption (`* 256`). The README table (â€œPixel MAE per corner (pixels)â€) is therefore mislabeled; deployment quality will not match reported numbers.

**Fix:** use per-sample `orig_size` (xÂ·W, yÂ·H), report both 256-space and original-pixel metrics, and state which one the table uses.

### P1-2 â€” `mobilenet_v3_large` is broken in both models (the config comments recommend it)
`MobileNetV3-Large.features` has 17 blocks; the hardcoded indices are Smallâ€™s.
- `point_regression.py` declares `feature_dims = [960, 160, 80, 40, 24]` for indices `[1,3,6,9,12]`; the real channels there are `16, 24, 40, 80, 160`.
- `heatmap_regression.py` declares `[24, 40, 80, 160]` for `[1,3,6,9]`; real: `16, 24, 40, 80`.

The first `nn.Conv2d(in_ch, â€¦)` projection will raise a channel-mismatch error. Only `mobilenet_v3_small` is tested. Configs (`point_base.yaml`, `heatmap_base.yaml`, QUICKSTART) advertise large as an option.

**Fix:** derive `feature_dims` by running a dummy tensor and reading `out.shape[1]`, instead of hardcoding; or remove the large branch.

### P1-3 â€” Config knobs that do nothing (and one that matters)
Ignored by `train_heatmap.py` / `train_point.py`:

`warmup_epochs`, `lr_schedule`, `gradient_clip`, `early_stopping_patience`, `save_best_only`, `log_interval`, all `augment_*` keys, `fpn_channels`, `num_bifpn_blocks` (the factory `create_heatmap_model()` does not even accept the last two â€” it silently uses 64/2).

Augmentations are hardcoded in `src/data/dataset.py` at fixed probabilities, so `configs/heatmap_phase3.yaml`â€™s â€œstronger augmentationâ€ section changes nothing. `heatmap_phase3.yaml` promises gradient clipping at 1.0; **no clipping happens anywhere in the heatmap trainer**.

**Fix:** either wire the keys through or delete them from configs; a config file that lies is worse than no config.

### P1-4 â€” Gradient clipping is skipped exactly where it is needed
`train_point.py` â†’ `train_one_epoch`: `clip_grad_norm_` is called only in the `scaler is None` branch (CPU). With AMP on CUDA â€” the normal case â€” gradients are never clipped. Correct AMP clipping additionally requires `scaler.unscale_(optimizer)` before clipping.

**Fix:** unscale â†’ clip â†’ step in the AMP path; add clipping to `train_heatmap.py`.

### P1-5 â€” Export/refine rebuild models without architecture params
`src/export.py`, `refine_corners.py`, `analyze_heatmap_model.py` all call `create_heatmap_model(backbone=â€¦, pretrained=False, with_classification=â€¦)`. `fpn_channels`/`num_bifpn_blocks` are dropped on the floor. It works only because every config happens to use the defaults. Any tuned config â†’ `load_state_dict` shape error, at export time, after hours of training.

`refine_corners.py` also reads architecture from `--config` (default `configs/heatmap_base.yaml`) instead of `checkpoint['config']`, so checkpoint/config drift is silent until it crashes. `src/export.py` takes `--img_size` from the CLI instead of the checkpoint config.

**Fix:** single `build_model_from_config(cfg)` helper used by train/export/refine/analyze; persist and always load `checkpoint['config']`.

### P1-6 â€” Classification head is dead code (and the artifact doesnâ€™t match the repo)
All three heatmap configs set `with_classification: true`, so the model builds a 2-layer head and returns `cls_logits` â€” but the dataset has no class label and the loss never consumes it. It trains nothing and costs compute.

Evidence of drift: the shipped `exports/heatmap_phase3.onnx` contains only the strings `image` and `heatmaps` (no `cls_logits`), i.e. it was produced by a model/config that differs from the current repo state. `exports/Program.cs` is byte-identical to `dotnet_project/Program.cs` (SHA-256 `CC9E9B0Dâ€¦`), a duplicate that will inevitably diverge.

**Fix:** remove the head (or implement the target); make `export.py` the only export path; delete `exports/Program.cs`.

### P1-7 â€” Augmented keypoints outside the frame become clamped lies
`CornerDataset` uses `A.KeypointParams(format='xy', remove_invisible=False)` then `np.clip(corners, 0, 1)`. `A.Perspective(scale=(0.05,0.15))` and `A.Affine(rotate=Â±15Â°, shear=Â±10Â°, scale 0.8â€“1.2)` routinely push corners outside the image. Those keypoints are not dropped and not marked; they are pinned to the border. The model is then explicitly trained to put a corner on the edge, indistinguishable from a real edge corner.

**Fix:** drop (retry) samples with any keypoint outside `[0,W]Ã—[0,H]`, or use `remove_invisible=True` plus filtering, and assert the corner count is still 4.

### P1-8 â€” Aspect ratio is destroyed by a square resize
All images (4:3 portrait at 3024Ã—4032) are squashed to 256Ã—256 (`A.Resize`), and inference does the same. Corner geometry in a 1.33 aspect image compressed to 1:1 is distorted, especially for perspective quad corners; this is a standard accuracy loss for document detection.

**Fix:** letterbox/pad to a square (and scale the points accordingly), or use a non-square input (e.g. 256Ã—320) with matching heatmap size.

### P1-9 â€” Random split leaks near-duplicate burst frames
`dataset/` filenames are burst sequences (`curved_0000â€¦`, `phone_*`). The split is a uniform `np.random.permutation` over sorted filenames, so consecutive near-identical frames land on both sides of the split. Validation metrics are optimistic.

**Fix:** group split by capture sequence/ID; keep a frozen split file in-repo.

### P1-10 â€” Absolute paths hardcoded to the old machine
- `image_labling.py`: `IMAGES_DIR = r"D:\Side Projects\Projects\PaperMask\images"` (does not exist; and `.jpg` only).
- `image_labling_validation.py`: `IMAGES_DIR`/`OUTPUT_DIR` under the same dead path.

As shipped, the annotator finds **0 images** and the validator prints â€œImage not foundâ€ for everything, then writes nothing. README tells users to run both.

**Fix:** `argparse` args with repo-relative defaults (`--images-dir dataset --labels labels.json`).

### P1-11 â€” Annotation tool cannot recover from a misclick, and zoom is unusable
`image_labling.py`:
- The 4th click immediately saves and advances; no undo, no confirm, no edit.
- `offset_x/offset_y` are declared but never modified â€” there is no panning. After zooming in (`+`), most of the image is outside the window and unreachable; `WINDOW_NORMAL` also lets the OS rescale the canvas, while `mouse_callback` maps raw window pixels through `(x+offset)/scale`, so clicks drift after any manual window resize.
- Hardcoded `*.jpg` only; no `--resume`/skip list (it re-walks every file).

**Fix:** add undo/backspace, require explicit save, implement pan (drag with middle/right button), handle `WINDOW_NORMAL` size via `cv2.getWindowImageRect`, and accept `jpg/jpeg/png`.

### P1-12 â€” Unsafe checkpoint loading
`torch.load(path, map_location='cpu')` in `src/export.py` (2Ã—), `refine_corners.py`, `analyze_heatmap_model.py` with PyTorch 2.0 defaults (`weights_only=False`) â†’ pickle deserialization = arbitrary code execution from any shared `.pth`. Checkpoints are 13â€“60 MB binaries that people routinely download/share.

**Fix:** migrate to `torch.load(..., weights_only=True)` (load config as plain dict), or store a `.safetensors`/JSON sidecar. Never load a checkpoint from an untrusted source.

### P2-1 â€” Training loop starves the GPU
`train_one_epoch`/`validate` call `pred_heatmaps.cpu().numpy()` and run `heatmap_to_points` (Python double loop over batchÃ—channels) **inside** the step, then `metrics.compute()` **every batch** (O(nÂ²) over the epoch) just to refresh the tqdm postfix. Each batch forces a device sync and Python-level work proportional to epoch length. Expect a large throughput loss versus the hardwareâ€™s capability.

**Fix:** keep metrics on GPU; compute postfix from a running mean; postprocess only on validation (or only first N batches).

### P2-2 â€” Heatmap targets are generated inefficiently
`_generate_heatmap` allocates a fresh `meshgrid` + `exp` per corner (4Ã— per sample), and `labels.json` is parsed twice per run (train + val dataset).

**Fix:** pre-compute/cache `(4,64,64)` targets per sample, or generate on GPU in a collate step; build the split once and share it.

### P2-3 â€” `denormalize_corners` mutates its input
`corners_np = corners.detach().cpu().numpy()` returns a view for CPU tensors; the subsequent in-place `*=` scales the callerâ€™s tensor. Used in `train_point.visualize_predictions`, so predictions are silently modified. Add `.copy()` or `clone()`.

### P2-4 â€” Forward-hook feature cache is a plain dict
`self.feature_maps = {}` in both models retains the last batchâ€™s activations (memory held until the next forward), and is not safe under `torch.jit.trace` / multi-threaded inference / `DataParallel`. Prefer registering buffers keyed by index, or better, restructure the backbone into an explicit `nn.Sequential` slice â€” which also makes ONNX export deterministic instead of hook-dependent.

### P2-5 â€” No reproducibility
`configs/*.yaml` exposes `seed`, but only `np.random.seed(seed)` is applied (a global side effect) for the split. `torch`, CUDA, DataLoader workers, and albumentations RNG are unseeded. Two runs with the same config are not comparable.

**Fix:** `torch.manual_seed`, `torch.cuda.manual_seed_all`, `worker_init_fn`, `generator=`, `torch.use_deterministic_algorithms(True)` (best-effort), and seed albumentations.

### P2-6 â€” .NET app resource/robustness issues
- `ProcessSingleFile`: `cropped` never disposed; `inputs`/results not disposed; `session` (static, `IDisposable`) never disposed.
- `ProcessFolder` silently writes same-basename outputs to one flat dir (collisions across subfolders), is non-recursive, and only looks for `*.jpg|jpeg|png`.
- All exceptions are reduced to `ex.Message` (no stack trace, no failed-file log), making batch failures undebuggable.
- Base-directory discovery walks up looking for a `.csproj`, so a published/standalone exe resolves paths incorrectly.
- `PredictCorners` writes ~65k values through `Mat.At<Vec3f>` per image; use `Marshal.Copy` on a continuous `Mat` for a large speedup.

### P2-7 â€” Documentation drift (a new reader is actively misled)
- `QUICKSTART.md` references `test_installation.py`, `debug_visualize.py`, `plan-modelDesign.prompt.md` â€” **none exist**.
- It says the dataset has **61 images**; it has 1,103 + 169, under different directories than documented (`images/` doesnâ€™t exist).
- It advertises a `--device cpu` flag: not implemented.
- README says labels may be clicked â€œin any orderâ€; training requires a fixed canonical order per channel (labels happen to be canonical, but nothing enforces it).
- README project tree omits `dataset_v2/`, `outputs/`, `dotnet_project/`, `torch_env/`.
- `refine_corners.py` defaults to `phase2_labels.json` while the repo ships `phase3_labels.json`; QUICKSTART says `refined_labels.json`.
- `requirements.txt` omits `scipy` (imported by `analyze_heatmap_model.py`) and pins `numpy==1.25.0`, `torch==2.0.1`, `Pillow==11.3.0`; `timm`/`scikit-image` are listed but unused.

### P2-8 â€” Repository hygiene
No `.gitignore`; `__pycache__` directories, `dotnet_project/bin|obj` (742 MB), `torch_env` (4.9 GB), datasets, checkpoints, and `exports/exports.zip` (a duplicate of its own directory) are all in-tree. 39,775 files for ~4,000 lines of first-party Python + 308 lines of C#. With no VCS there is no diff history to review or revert.

**Minimum `.gitignore`:** `torch_env/`, `__pycache__/`, `*.pyc`, `dataset*/`, `outputs/`, `exports/*.onnx`, `exports/*.zip`, `dotnet_project/bin/`, `dotnet_project/obj/`, `.vs/`.

### P2-9 â€” Unused/commented-out code and traps
- `train_point.py` duplicates `canvas[:, img_size*2:] = overlay` (harmless, signals copy-paste).
- `analyze_heatmap_model.py` imports `scipy` and hardcodes `labels.json`/`dataset`/batch=1 rather than taking CLI args; its â€œ/100 scoreâ€ is an invented heuristic presented like an evaluation.
- `AdaptiveWingLoss` builds `self.C` from a CPU `torch.tensor` and uses tensor-valued exponents (`alpha - target`); it is unused (configs use `weighted_mse`) but remains exported and importable.
- `FocalHeatmapLoss` clamps already-sigmoided predictions (no-op) and its â€œfocalâ€ weight `2Â·|err|Â²` is not focal loss in the usual sense.
- `src/losses/__init__.py` / `src/metrics/__init__.py` are effectively empty stubs while `src/models/__init__.py` re-exports; inconsistent.

### P2-10 â€” No test suite
No smoke test for: dataset loading, model forward shapes, ONNX export, postprocess, metrics, or .NET decode parity. Given P0-1/P0-3/P1-2, a 30-line test file would have caught every one of them.

---

## 3. Verified facts (appendix)

```
labels.json         169 keys  â†” dataset_v2 (169 jpg)  â€” 0 missing, 0 extra, 0 malformed
phase3_labels.json 1103 keys  â†” dataset   (1103 jpg) â€” 0 missing, 0 extra, 0 malformed
                     all 4-point, unique, clockwise, convex, non-self-intersecting
sample image        dataset/curved_0000.jpg = 3024 Ã— 4032
                    dataset_v2/0169.jpg     = 1536 Ã— 2048
checkpoints         outputs/heatmap_phase3/{best,last}.pth   12.99 MB each
                    outputs/point_regression/{best,last}.pth 59.67 MB each
ONNX                exports/heatmap_phase3.onnx 1.53 MB â€” strings: "image" Ã—2, "heatmaps" Ã—4
                    (no "cls_logits", no "edges")
Program.cs          exports/Program.cs == dotnet_project/Program.cs (identical hash)
csproj              net9.0; Microsoft.ML.OnnxRuntime 1.17.0; OpenCvSharp4 4.8.1.0
```

---

## 4. Recommended order of work

1. **Stop the bleeding (minutes):** add `.gitignore`, `git init` + first commit, delete `torch_env/` from tracking, rebuild a working venv, make `--labels_path`/`--images_dir` required and validated (P0-1, P0-2, P2-8).
2. **Make the numbers honest (hours):** per-axis original-resolution metrics; relabel the label files as pseudo-labels; freeze a human-labeled held-out test set (P0-4, P1-1). Until then, do not quote MAE/IoU as accuracy.
3. **Make training feasible (hours):** fix the point-model attention (P0-3), derive feature dims at runtime or drop large (P1-2), wire or delete dead config keys, add clipping (P1-3, P1-4).
4. **Make deployment match training (day):** sub-pixel decode + quad validation + clamping in C#, one export path, ONNX/PyTorch/C# parity test on 20 images (P0-5, P1-5, P1-6).
5. **Then:** letterbox resize, group split, GPU-side metrics, arg-parsing for tools, tests/CI (P1-7â€¦P1-9, P2-1â€¦P2-10).

---

## 5. Corrected run commands

```powershell
# Fresh environment (the committed torch_env is broken)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install scipy                      # imported by analyze_heatmap_model.py

# Heatmap training â€” note the labels/images pair (they must match)
python src/train_heatmap.py `
  --config configs/heatmap_phase3.yaml `
  --labels_path phase3_labels.json `
  --images_dir dataset `
  --output_dir outputs/heatmap_phase3_run

# Point training (currently OOM-prone; see P0-3)
python src/train_point.py `
  --config configs/point_base.yaml `
  --labels_path phase3_labels.json `
  --images_dir dataset `
  --output_dir outputs/point_run

# .NET inference
dotnet run --project dotnet_project
```

# PaperMask v2 — Lite-FPN corner detector with sub-pixel offset head

Date: 2026-10-28
Status: draft for review
Branch: `feat/litefpn-heatmap-offset`
Supersedes: the MobileNetV3 + BiFPN heatmap model and the cross-attention point model, plus the
`dataset/` + `dataset_v2/` pseudo-labelled data.

## 1. Goal

Retrain PaperMask on the synthetic `papermask` split of `walidmostafa77/nexadoc-50k` with a single,
exportable architecture: MobileNetV3-Small trunk, Lite-FPN neck, heatmap head plus a sub-pixel
offset head.

The output must be one ONNX file that a .NET application can run on CPU and that produces corners
accurate to well under one pixel at 256×256 — closing the 4-px quantization gap the current C#
decoder has.

### Success criteria

| Criterion | Target | How measured |
|---|---|---|
| Corner MAE (256-space) | ≤ 1.0 px | test split, offset decoding |
| Quad IoU | ≥ 0.985 | test split |
| PCK @ 1 % | ≥ 0.98 | test split |
| ONNX vs PyTorch | max abs diff ≤ 1e-4 | same batch, `onnxruntime` |
| C# vs Python decode | ≤ 0.5 px per corner | 20-image parity fixture |
| CPU latency (1 thread) | ≤ 40 ms @ 256×256 | `dotnet` console timing |
| Train time | ≤ 8 h on 2×T4 | Kaggle notebook |

Targets are calibrated to the synthetic data: corners are exact geometry, so anything above ~1 px
means a code or decoding bug, not a modelling limit.

### Non-goals

- No mask segmentation output (the `papermask` task is corners only).
- No change to `image_labling.py` / `image_labling_validation.py`; they serve the retired
  human-labelling workflow and stay untouched (documented as legacy).
- No change to `deshadow`, `dewarp` or `upsample` tasks.
- No re-rendering: the Kaggle dataset is the source of truth.

## 2. Data contract

Source: `nexadoc-50k`, 7.8 GB zip, extracted by Kaggle under `/kaggle/input/nexadoc-50k/`.

```
<root>/papermask/
  train/images/NNNNN.jpg   labels.json   (16,032 images)
  val/images/NNNNN.jpg     labels.json   ( 1,961 images)
  test/images/NNNNN.jpg    labels.json   ( 2,007 images)
```

`labels.json`:
```json
{"00000.jpg": [[0.41, 0.12], [0.93, 0.15], [0.95, 0.88], [0.39, 0.90]]}
```
- 4 points, normalized `[0,1]`, rounded to 6 dp.
- Order TL, TR, BR, BL, image origin top-left.
- Images are 256×256 with the page filling 60–70 % of the frame.

`data_root` resolution: if `<data_root>/papermask` exists, use it; else if
`<data_root>/images` and `<data_root>/labels.json` exist in the split dirs, use `<data_root>`
directly. Both layouts are accepted because the local `runs/*` output and the zip root differ;
the resolver prints the path it picked.

### Preflight (hard gate, runs at dataset construction)

Fails with a precise message and a count when:

1. `<split>/images` or `<split>/labels.json` is missing.
2. A label key has no corresponding image file (reports up to 5 names; any missing = abort).
3. An image has no label entry (warn only — unused images are not an error).
4. A quad is not exactly 4 points, has a duplicate point, or has area below 1e-4.
5. Any coordinate is outside `[-0.001, 1.001]`.
6. `labels.json` is not a flat `dict[str, list[list[float]]]` — this is the check that catches the
   legacy `[{"value": ..., "Count": 2}]` shape if the Kaggle artifact predates the current
   generator.

Images are *not* re-validated pixel by pixel (20k decodes at load time is wasteful); `cv2.imread`
returning `None` during training raises instead of substituting a black image.

### Augmentation

Applied to the 256×256 image with corners in pixel space, then converted back to normalized.

| Transform | Range | p |
|---|---|---|
| HorizontalFlip | — | 0.5 (corners reordered BR,BL,TL,TR) |
| RandomBrightnessContrast | 0.2 | 0.5 |
| HueSaturationValue | 0.15 | 0.3 |
| GaussNoise | var 10–30 | 0.2 |
| Affine (rotate / scale / shear) | ±8° / 0.95–1.05 / ±3° | 0.5 |
| Perspective | 0.03 | 0.3 |

After the geometric transforms, if any corner falls outside `[0, 256]²` in pixel space the sample
is **rejected and re-drawn** (up to 5 attempts, then skipped). No clamping: a corner pinned to the
border is a false label.

Val/test transforms: none.

## 3. Model

`src/models/paper_corner.py`

```
input  (B,3,256,256), ImageNet-normalized RGB
  └─ mobilenet_v3_small.features  (torchvision, optional pretrained weights)
       taps: idx 1 → 16ch /4   idx 3 → 24ch /8   idx 6 → 40ch /16   idx 9 → 96ch /32
  └─ LiteFPN(C=64)
       lat[i] = 1×1 Conv(c_i → C) + BN + SiLU
       td[3] = lat[3]
       td[i] = DWConv3x3(lat[i] + Up(td[i+1]))          # i = 2,1,0; Up = nearest ×2
       out[i] = DWConv3x3(td[i]) + lat[i]               # residual
  └─ head on out[0]  (64×64)
       stem   = DWConv3x3(64→64) + SiLU
       heatmaps = 1×1 Conv(64→4)  → sigmoid        (B,4,64,64)
       offsets  = 1×1 Conv(64→8)                   (B,8,64,64)
```

- `heatmap_size = 64`, `stride = 4`.
- Heatmap bias initialised to `-4.6` (focal prior p≈0.01); offset conv zero-initialised.
- Params ≈ 1.1 M. Backbone weights come from `torchvision`, so `timm` is no longer needed.
- Taps are computed with an explicit `nn.ModuleList` slice, not forward hooks — this removes the
  mutable `feature_maps` dict and makes ONNX tracing deterministic.

Backbone name, pretrained flag and FPN width are config-driven. A `build_model(cfg)` factory is
the only construction path, used by training, export, evaluation and refinement (fixes P1-5).

## 4. Targets, loss, decoding

### Encoding (256 px → 64 cells)

```
hm = p_px / 4                     # corner in heatmap coordinates
cell = floor(hm)                  # clamped to [0, 63]
off  = hm - cell                  # ∈ [0,1)
gaussian = exp(-((x-cell_x-? )…)) # centred on hm, σ = 2.0 cells
```
Concretely, target heatmap for corner *c* is `exp(-((X - hm_x)² + (Y - hm_y)²) / (2σ²))` on the
64×64 grid; offset target is `off` at `(cell_y, cell_x)` only.

### Loss

```
L_hm = -1/N Σ [ (1-p)^2 · log(p) · y        +   (1-y)^4 · p^2 · log(1-p) · (1-y) ]
L_off = 1/N Σ |off_pred - off_gt|          # gathered at the 4 GT cells only
L = 1.0 · L_hm + 1.0 · L_off
```
`N = B×4`, `y` = Gaussian target, `p` = sigmoid output — the standard penalty-reduced focal loss.

`src/losses/heatmap_focal.py` and `smooth_l1.py` are replaced by one `corner_loss.py` implementing
exactly the above.

### Decoding (used by validation, refinement and the C# port)

```
for each corner channel c:
    (iy, ix) = argmax(hm[c])            # 64×64
    score    = hm[c, iy, ix]
    x_px = (ix + off[2c,   iy, ix]) * 4
    y_px = (iy + off[2c+1, iy, ix]) * 4
```
No `+0.5`: the offset target is defined relative to `floor`, so the pair is self-consistent and
exact for a perfect prediction.

Post-decode validation: clamp to `[0,256]`, require convex and non-degenerate (area ≥ 64 px²), and
normalise winding to clockwise with corner 0 = top-left. Confidence = min of the four scores.

## 5. Metrics

`src/metrics/corners.py`, all in 256-space (the native resolution of both training and inference):

- MAE and RMSE per corner in pixels (mean over corners and samples).
- Normalized MAE (÷ 256).
- PCK @ 1 %, 2 %, 5 % of the image diagonal.
- Quad IoU (shapely polygon intersection; zero for invalid quads).
- Per-corner breakdown and the fraction of quads failing validation.

Decoding happens once per validation epoch on GPU tensors; only the final 5 scalars are synced to
the host. No `.cpu()` inside the training loop, no `compute()` per batch (fixes P2-1).

## 6. Training

`src/train.py`, single entry point.

```
python src/train.py --config configs/paper_base.yaml [--resume ckpt.pth]
```

Everything is read from the config; unknown keys are a hard error (`ConfigError`), so a knob that
does nothing is impossible (fixes P1-3). Implemented keys:

```yaml
seed: 42
data:   {root: /kaggle/input/nexadoc-50k, img_size: 256, num_workers: 4, augment: true}
model:  {backbone: mobilenet_v3_small, pretrained: true, fpn_channels: 64,
         heatmap_size: 64, sigma: 2.0}
train:  {batch_size: 32, epochs: 100, lr: 3.0e-4, weight_decay: 2.0e-4,
         warmup_epochs: 3, grad_clip: 1.0, amp: true, early_stop_patience: 15,
         save_best_only: true, log_interval: 50}
loss:   {heatmap_weight: 1.0, offset_weight: 1.0}
```

Loop requirements:

- Seed `random`, `numpy`, `torch`, `torch.cuda`, DataLoader workers, albumentations.
- AMP with the correct order: `scaler.scale(loss).backward()` → `scaler.unscale_(opt)` →
  `clip_grad_norm_(1.0)` → `scaler.step` → `scaler.update()`. Clipping therefore also applies on
  GPU (fixes P1-4).
- LR: linear warmup over `warmup_epochs`, then cosine to 1 % of peak.
- Checkpoints: `{arch_version: 2, config: <dict>, state_dict, epoch, metrics}`. `best.pth` per
  validation MAE; `last.pth` every epoch unless `save_best_only`.
- Early stopping on val MAE with `early_stop_patience`.
- TensorBoard scalars; per-epoch visualisation grid (image, GT, predicted heatmap, decoded quad)
  for the first 8 val samples.

## 7. Export

`src/export.py`

```
python src/export.py --checkpoint outputs/paper_run/best.pth --output-dir exports/paper_v2
```
- Rebuild with `build_model(checkpoint['config'])`, `load_state_dict`, `eval()`.
- `torch.onnx.export`, opset 17, dynamic batch axis only.
- Input `image` (B,3,256,256); outputs `heatmaps` (B,4,64,64) and `offsets` (B,8,64,64).
- Self-check: run the traced graph in `onnxruntime` on the same batch and assert
  `max|diff| ≤ 1e-4`; fail the export otherwise.
- Write `model.json` next to the `.onnx`:

```json
{"arch_version": 2, "corner_order": "TL,TR,BR,BL", "confidence_threshold": 0.3,
 "input":  {"name": "image", "height": 256, "width": 256, "layout": "CHW", "color": "RGB",
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
 "output": {"heatmaps": "heatmaps", "offsets": "offsets", "heatmap_size": 64, "stride": 4}}
```

`refine_corners.py` and the evaluation script load the same `model.json`; there is exactly one
export path.

## 8. .NET inference

`dotnet_project/Program.cs`

- Load `model.json`; no hardcoded tensor names, sizes or paths relative to `.csproj`.
- Preprocess unchanged (resize to 256, BGR→RGB, ImageNet mean/std, CHW).
- Decode with offsets as in §4; confidence gate from `model.json`.
- Validate: convex, clockwise, area ≥ 64 px², clamp to source bounds; on failure log the file and
  skip it (per-image failure must not abort a folder run).
- Recursive folder scan, output path mirrors the input subdirectory (no basename collisions).
- `using`/`Dispose` for every `Mat`, `DenseTensor` and the `InferenceSession`; stack traces on
  error.

## 9. Kaggle

Follows the `DocShadow-Lite/notebook/doc-shadow-lite.ipynb` pattern, which is already proven on
this account: **clone from GitHub, patch a Kaggle copy of the YAML, train, export, package**.

- `kaggle/papermask.ipynb` — 14 cells:
  1. silence TF/warnings, uninstall TensorFlow, `pip install 'protobuf<4' albumentations onnx onnxruntime`,
     then `git clone --depth 1 --branch main https://github.com/walidmoustafa2077/PaperMask.git /kaggle/working/PaperMask`
     (idempotent `rmtree` first).
  2. load `configs/paper_base.yaml`, patch only `data.root → /kaggle/input/nexadoc-50k`,
     `train.output_dir → /kaggle/working/runs/papermask`, `batch_size 64`, `epochs 120`,
     `lr 6e-4`, `amp true`; write `configs/kaggle_paper.yaml`. Architecture keys are untouched.
  3. preflight: `src/train.py --preflight` validates labels/images before any GPU time.
  4. train, export (`best.pth` → `paper_v2.onnx` + `model.json`), evaluate on `test`.
  5. package: zip `artifacts/`, visualisations, and `best.pth` only; `FileLink` for each.
- No pre-resize cache step (unlike DocShadow): inputs are already 256×256, so decode is ~4 % of a
  step and a cache would add 600 MB of writes and a staleness bug.
- Resume: re-run cells 1–2, then `--resume /kaggle/working/runs/papermask/last.pth`.
- `kaggle/README.md` — inputs, accelerator (T4×2, internet on), expected runtimes, what to download.

Repo requirement: **the GitHub repo must be public**, because the notebook clones it without a
token. If it ever goes private, the clone URL needs a PAT and `isInternetEnabled` stays on.

## 10. Tests

`tests/` (pytest, no network):

| Test | Asserts |
|---|---|
| `test_dataset_contract` | 3-image fixture loads; shapes, dtypes, normalized ranges, split isolation |
| `test_dataset_rejects_missing_image` | raises, message names the file |
| `test_dataset_rejects_bad_labels` | 3 points / duplicate / out-of-range / legacy dict shape all raise |
| `test_model_forward` | (1,3,256,256) → (1,4,64,64) and (1,8,64,64); heatmap in (0,1) |
| `test_encode_decode_roundtrip` | random corners → targets → perfect logits → error < 0.5 px |
| `test_loss_perfect_prediction` | focal + offset loss ≈ 0 |
| `test_config_schema` | unknown key raises; missing required key raises |
| `test_onnx_parity` | export a 1-batch model, compare outputs (marked `slow`) |

## 11. Migration / deletions

Removed:

```
src/models/point_regression.py
src/models/heatmap_regression.py
src/train_point.py
src/train_heatmap.py
src/data/dataset.py
src/losses/smooth_l1.py
src/losses/awing.py
src/losses/heatmap_focal.py
configs/point_base.yaml
configs/heatmap_base.yaml
configs/heatmap_phase2.yaml
configs/heatmap_phase3.yaml
analyze_heatmap_model.py
```

Rewritten: `src/export.py`, `src/metrics/geometry.py` → `corners.py`, `refine_corners.py`,
`dotnet_project/Program.cs`.

Added: `src/models/paper_corner.py`, `src/models/litefpn.py`, `src/data/paper_dataset.py`,
`src/losses/corner_loss.py`, `src/metrics/corners.py`, `src/config.py`, `src/train.py`,
`src/evaluate.py`, `configs/paper_base.yaml`, `tests/*`, `kaggle/*`.

Kept for now: `labels.json`, `phase3_labels.json`, `dataset*/`, `outputs/` (untracked, local
reference only). Old checkpoints are incompatible with arch v2 — this is intentional; they are not
in git.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Kaggle labels are the legacy `[{"value":...}]` shape | preflight check #6 fails fast with a sample; conversion is a small adapter if needed |
| `papermask` upload predates the 4-corner contract | same as above; verify on 1 image before the full run |
| V3-Small + 64 ch is capacity-limited | `fpn_channels` is config; if MAE plateaus > 1.5 px, try 96 or `mobilenet_v3_large` (fixed features) |
| Offset head learns background noise | offset loss masked to the 4 GT cells only |
| ONNX export of `nearest` + depthwise ops | parity test in export + pytest; opset 17 |
| 20k samples with 5× augmentation attempts | rejection sampling capped, logged if > 5 % skipped |
| Kaggle session limits (9–12 h) | `--resume`, checkpoints every epoch, `save_best_only` |

## 13. Open items (to close before implementation)

1. Approve the architecture and deletion list above.
2. ~~Confirm the Kaggle code-upload route~~ — resolved: `gh` is authenticated as
   `walidmoustafa2077`, so the notebook clones the repo exactly like DocShadow-Lite does.
3. After the first preflight run on Kaggle, confirm the label shape and report the actual
   `data_root` path printed by the resolver.

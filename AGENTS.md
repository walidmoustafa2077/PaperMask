# PaperMask — Agent Handoff

Read this first. It is the compressed state of the repo.

History: `AUDIT.md` (the original audit of the pre-2026 codebase) and
`docs/superpowers/specs/2026-10-28-papermask-litefpn-retrain-design.md` +
`docs/superpowers/plans/2026-10-30-papermask-litefpn-implementation.md`.
The audit describes a codebase that no longer exists; treat it as background,
not as current truth.

## What this is

Detect the four corners of a photographed page, then perspective-warp it flat.

One model: MobileNetV3-Small backbone → Lite-FPN neck (64 ch, 4 levels at
strides 4/8/16/32) → two heads:

| Head | Shape | Notes |
|---|---|---|
| heatmap | (B,4,64,64) | sigmoid, bias init −4.6, Gaussian σ=2 cells |
| offsets | (B,8,64,64) | one (dx,dy) per corner, zero-init, in cells |

~0.33 M params. Decode is `(argmax + offset) * 4` → sub-pixel corners in input
pixels. Corner order is TL, TR, BR, BL in normalized coordinates everywhere
except the model input/output, which is pixels.

Deployment: `src/export.py` → `paper_corner.onnx` + `model.json` →
`dotnet_project/Program.cs` (.NET 9, OnnxRuntime 1.17, OpenCvSharp 4.9).

## Map

```
src/config.py              strict YAML schema — unknown keys raise ConfigError
src/train.py               --config --preflight --resume
src/evaluate.py            --config --checkpoint --split
src/export.py              --checkpoint --output-dir, parity-gated (1e-4)
src/data/paper_dataset.py  resolve_data_root, preflight, order_corners, augmentation
src/models/litefpn.py      DepthwiseSeparableConv, LiteFPN
src/models/paper_corner.py build_model (runtime channel discovery)
src/losses/corner_loss.py  encode_targets, CornerLoss (focal + L1)
src/metrics/corners.py     MAE/RMSE/PCK@1,2,5%/IoU via shapely
src/utils/corners.py       decode_corners, quad_props, winding
src/utils/preprocess.py    letterbox (scale-to-fit + grey pad) + point mapping
src/utils/framing.py       reframe: random scale/position, labels follow
src/data/make_framing.py   bake reframed samples into a static dataset
refine_corners.py          --checkpoint --images-dir --output [--auto]
image_labling*.py          legacy pixel-space labelers (do not extend)
dotnet_project/            PaperMask.Console.csproj — build with the .csproj path
kaggle/papermask.ipynb     cloud training (clones this public repo)
configs/paper_base.yaml    the only config
tests/                     pytest (42 fast + 4 slow)
```

## Environment reality

- Use `.venv` (torch 2.4.1+cpu, albumentations 2.0.8, onnxruntime 1.31.1).
  `torch_env/` is a dead committed venv — never use it, never commit another.
- `dotnet build dotnet_project` fails with MSB1011 (a .sln sits next to the
  .csproj). Always `dotnet build dotnet_project/PaperMask.Console.csproj`.
- `git` warns about LF→CRLF on every commit. Harmless.
- Dataset: `data/nexadoc-50k` is **not** on this machine. Only
  `data/smoke/papermask` (12 synthetic 256×256 images) exists, for tests.
- Training happens on Kaggle (`kaggle/papermask.ipynb`), not locally.

## Data contract

```
<root>/papermask/{train,val,test}/images/NNNNN.jpg
<root>/papermask/{train,val,test}/labels.json   {"00000.jpg": [[x,y]×4]} normalized
```

Splits are produced by the generator and must **never** be re-shuffled. The
loader accepts `<root>/papermask` or a root that already contains `train/`,
`val/`, `test/`.

## Five things that will bite you

1. **Do not re-split.** Fixed 80/10/10 by index is what keeps near-identical
   renders out of validation.
2. **`torch.onnx.export` leaves the module in training mode.** `src/export.py`
   calls `wrapper.eval()` twice and computes parity under `no_grad`. If you
   refactor export, keep that or BatchNorm will silently break parity.
3. **Augmentation rejects, it does not clamp.** A quad that leaves the frame
   causes a retry, then falls back to the clean image. Never reintroduce
   `np.clip` on keypoints.
4. **The C# side mirrors the Python preprocessing** — but it still centre-crops.
   Python now letterboxes (`src/utils/preprocess.py`): scale-to-fit + grey pad,
   geometry preserved. The .NET app takes the largest centred square crop,
   which throws away up to half of a tall photo and is measurably worse
   (`docs/reports/2026-11-01-real-photo-evaluation.md`). Port `letterbox` to C#
   as the last step, then re-run the parity check.
5. **Framing augmentation is the fix for real photos, not preprocessing.**
   `data.framing: true` rescales each sample 0.55–1.35× and re-places it, so the
   model stops assuming a square frame with the page at 60–70 %. Leave it on.
   `src/data/make_framing.py` bakes the same op to disk if you want it static.

## Verified commands

```powershell
.\.venv\Scripts\Activate.ps1
pytest                      # 42 passed
pytest -m slow              # 4 passed: ONNX parity + `-m` entry points
python -m src.train --config configs/paper_base.yaml --preflight
dotnet build dotnet_project/PaperMask.Console.csproj
```

## Numbers from the last smoke run (synthetic, meaningless for quality)

Trained 2 epochs on 8 images: val MAE 123.6 px, then dropped the export
artifact (untrained model, low confidence) — expected.

## Open items

1. A real Kaggle run **has** happened (`nexadoc-50k`, 120 epochs, val MAE
   0.28 px, PCK@1 % 0.997, IoU 0.993) — but only on the uniform synthetic
   framing, and it is weak on real photos. See
   `docs/reports/2026-11-01-real-photo-evaluation.md`. The next run must be
   with `data.framing: true` (now the default in `configs/paper_base.yaml`).
2. `tests/parity/make_fixture.py` does not exist yet; C# ↔ Python parity was
   verified by hand on the smoke model (`tests/parity/README.md` describes the
   intended fixtures).
3. `confidence_threshold` (0.3 from `model.json`) is enforced in C# only, not
   in `evaluate.py` — deliberate, so recall loss stays visible.
4. Legacy `image_labling*.py` still have hardcoded absolute paths and write
   pixel coordinates. `refine_corners.py` is the supported tool.
5. The .NET app still centre-crops instead of letterboxing (item 4 above) and
   still ships `paper_corner.onnx` built by the previous run.

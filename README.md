# PaperMask

Detect the four corners of a photographed document, then perspective-warp it flat.

One small model does the whole job: a MobileNetV3-Small feature extractor, a
Lite-FPN neck, and two heads — a 4-channel heatmap (one per corner) plus a
2-channel-per-corner sub-pixel offset. Inference decodes
`(argmax + offset) * stride`, so corner precision is not limited by the 64×64
heatmap grid. The exported ONNX runs in the .NET console app in ~1 MB.

| | |
|---|---|
| Input | 256×256 RGB, ImageNet normalization |
| Heatmap | 4×64×64, stride 4, Gaussian σ = 2 cells |
| Offsets | 8×64×64, one (dx, dy) per corner |
| Parameters | ~0.33 M |
| Corner order | TL, TR, BR, BL |
| Output | `paper_corner.onnx` + `model.json` |

## Layout

```
src/
  config.py             strict YAML schema (unknown keys are an error)
  train.py              training loop, AMP, warmup+cosine, checkpoints
  evaluate.py           metrics on a checkpoint + split
  export.py             checkpoint -> ONNX, parity-gated, writes model.json
  data/paper_dataset.py split discovery, preflight, augmentation
  models/litefpn.py     depthwise-separable top-down FPN
  models/paper_corner.py backbone taps -> neck -> heatmap/offset heads
  losses/corner_loss.py CenterNet focal loss + masked L1 on offsets
  metrics/corners.py    MAE/RMSE/PCK/IoU
  utils/corners.py      target decoding and quad geometry
  utils/preprocess.py   letterbox: scale-to-fit + grey pad (shared geometry)
  utils/framing.py      reframe: random scale/position augmentation
  data/make_framing.py  materialise reframed samples as a static dataset
refine_corners.py       predict -> drag -> write labels (GUI)
image_labling.py        legacy click labeler (pixel coords, see note)
dotnet_project/         ONNX inference + warp console app
kaggle/papermask.ipynb  end-to-end cloud training notebook
configs/paper_base.yaml the only config
tests/                  pytest
```

## Data

Training data comes from the synthetic document generator
(`nexa-doc-dataset`), published as
[kaggle.com/datasets/walidmostafa77/nexadoc-50k](https://www.kaggle.com/datasets/walidmostafa77/nexadoc-50k).
Only the `papermask` task is used: 20,000 samples, already split 80/10/10.

```
<root>/papermask/{train,val,test}/images/NNNNN.jpg
<root>/papermask/{train,val,test}/labels.json
```

`labels.json` maps a filename to four normalized corners:

```json
{"00000.jpg": [[0.21, 0.34], [0.88, 0.30], [0.91, 0.95], [0.18, 0.92]]}
```

The loader accepts `<root>/papermask` or a root that already contains the split
directories. Splits are never re-shuffled — that is what prevents leakage
between near-identical renders.

`--preflight` validates the whole dataset before a run: image/label pairing,
exactly four points, coordinates in range, positive area, and it rejects the
legacy `[{"value": …, "Count": 2}]` label shape with a clear message.

## Framing

The generated set is uniform: every frame is square and the page covers
60–70 % of it. Real photos are 1:1 to 1:2.3, with the page anywhere from tiny
to bleeding off the edge. A model trained only on the uniform framing moves
its answer by 30–90 px when the framing changes.

Two mechanisms fix that:

- **Letterbox preprocessing** (`src/utils/preprocess.py`) — non-square input is
  scaled to fit and padded with flat grey instead of being squashed into a
  square. Squashing stretches the page and moves corners to places the model
  has never seen. `refine_corners.py` and the dataset both use this; the .NET
  app still centre-crops and should be switched to match.
- **Framing augmentation** (`src/utils/framing.py`, `data.framing: true`) — each
  training sample is rescaled 0.55–1.35×, dropped at a random position and
  cropped where it overflows, with the corner labels transformed by the same
  affine. Corners that leave the frame cause a retry, never a clamp.

`src/data/make_framing.py` bakes the same transformation into a static dataset
if you want something inspectable on disk:

```powershell
python -m src.data.make_framing --source data/nexadoc-50k/papermask `
  --dest data/papermask_frames --seed 42
```

On-the-fly framing gives a fresh draw every epoch, so it is the default; the
materialiser exists for reproducible datasets and inspection.

## Train

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt

python -m src.train --config configs/paper_base.yaml --preflight
python -m src.train --config configs/paper_base.yaml
python -m src.train --config configs/paper_base.yaml --resume runs/paper_base/last.pth
```

Always use `-m`. `python src/train.py` puts `src/` itself on `sys.path`, so the
`from src.…` imports fail with `ModuleNotFoundError: No module named 'src'`.

Checkpoints are `{arch_version: 2, config, state_dict, optimizer, scheduler, epoch, metrics}`.
`last.pth` is written every epoch, `best.pth` whenever validation MAE improves.
The architecture is always rebuilt from the checkpoint's own config, so a
checkpoint can never be loaded into the wrong model.

On Kaggle use `kaggle/papermask.ipynb` — it clones this repo, points the config
at `/kaggle/input/nexadoc-50k`, trains, exports, evaluates and packages the
artifacts.

## Evaluate and export

```powershell
python -m src.evaluate --config configs/paper_base.yaml --checkpoint runs/paper_base/best.pth --split test
python -m src.export --checkpoint runs/paper_base/best.pth --output-dir exports
```

`export.py` refuses to leave a broken artifact behind: it runs the ONNX graph
against the PyTorch reference and deletes the file if `max |diff| > 1e-4`.
`model.json` carries everything the C# side needs (input names/size, mean/std,
heatmap size, stride, corner order, confidence threshold).

## Refine / label

```powershell
python refine_corners.py --checkpoint runs/paper_base/best.pth `
  --images-dir data/nexadoc-50k/papermask/val/images --output refined_labels.json
```

Model prediction → drag corners → normalized labels, same format as
`labels.json`. `--auto` writes predictions without opening a window.

## Run the .NET app

```powershell
dotnet build dotnet_project/PaperMask.Console.csproj
dotnet run --project dotnet_project/PaperMask.Console.csproj -- path\to\paper_corner.onnx
```

Then type an image path, a folder (recursive), or `q`. Returns detected corners,
rejects low-confidence or non-convex quads instead of warping garbage, and writes
flattened images. Model resolution order: CLI argument → `PAPERMASK_MODEL` env →
`exports/paper_corner.onnx`. `PAPERMASK_DEBUG_DECODE=1` prints raw decoded corners.

## Tests

```powershell
pytest            # fast
pytest -m slow    # ONNX export + parity
```

## Note on `image_labling.py`

This is the original click labeler and it still writes **pixel** coordinates in
the pre-2026 format. It is kept for reference only; the current contract is
normalized coordinates, which `refine_corners.py` produces.

## License

MIT — see `LICENSE`.

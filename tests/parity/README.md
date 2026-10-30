# C# ↔ Python parity fixtures

The console app must reproduce the Python decoder exactly. `src/utils/corners.py::decode_corners`
and `dotnet_project/Program.cs::Decode` are the same arithmetic:

```
corner = (argmax_index + offset) * stride
```

## Why fixtures are generated, not committed

A parity fixture is only meaningful for a *specific* model. Generating them requires a trained
checkpoint, which requires a Kaggle run, so the fixtures are produced after the first export and
are not checked in (they would be ~20 × 256 × 256 × 3 floats).

## Generating

After a training run, on the machine that has the checkpoint:

```powershell
python tests/parity/make_fixture.py `
  --checkpoint outputs/paper_base/best.pth `
  --onnx exports/paper_v2.onnx `
  --data data/nexadoc-50k `
  --count 20 `
  --out tests/parity/fixtures
```

This writes, per sample, the image path and the corners predicted by PyTorch/ONNX from Python
(`corners.json`). Then:

```powershell
dotnet run --project dotnet_project -- tests/parity/fixtures/00000.jpg
```

and compare the printed corners against `corners.json`. Tolerance: **0.5 px** in 256×256 space.

## What is covered without a checkpoint

`Decode` and `ValidateQuad` are pure functions over float arrays, so the arithmetic is verified
by unit tests that do not need a model:

- peak at cell `(ix, iy)` with zero offset → `(ix * stride, iy * stride)`
- peak with offset `(0.5, 0.5)` → half a cell further
- negative and out-of-frame corners rejected
- counter-clockwise quads rejected
- area below 64 px² rejected

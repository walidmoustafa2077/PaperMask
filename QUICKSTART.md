# Quickstart

Full documentation is in `README.md`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt

# 1. Validate the dataset layout before spending GPU time
python -m src.train --config configs/paper_base.yaml --preflight

# 2. Train
python -m src.train --config configs/paper_base.yaml

# 3. Evaluate on the untouched test split
python -m src.evaluate --config configs/paper_base.yaml `
  --checkpoint runs/paper_base/best.pth --split test

# 4. Export ONNX (+ model.json), parity-checked
python -m src.export --checkpoint runs/paper_base/best.pth --output-dir exports

# 5. Run the .NET app
dotnet run --project dotnet_project/PaperMask.Console.csproj -- exports\paper_corner.onnx
```

Training on Kaggle instead: open `kaggle/papermask.ipynb`. It needs the
`walidmostafa77/nexadoc-50k` dataset attached and a GPU + internet enabled.

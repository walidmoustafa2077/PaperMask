# Quick Start: Corner Refinement Workflow

## Step 1: Test the Refinement Tool

Try it with your existing validation images first:

```powershell
# Activate environment
.\torch_env\Scripts\activate

# Run interactive refinement
python refine_corners.py --checkpoint outputs/heatmap_regression/best.pth
or 
python refine_corners.py --checkpoint outputs/heatmap_phase2_v2/best.pth
```

**What you'll see:**
- OpenCV window opens with first image
- Model predictions shown as green corners with numbers (1-4)
- Green quadrilateral connecting corners

**Try the controls:**
1. **Click-and-drag** a corner to adjust it
2. Press **r** to reset to original prediction
3. Press **u** to undo your last move
4. Press **n** to save and move to next image
5. Press **q** when done to save all and quit

## Step 2: Understand the Output

After refinement, you'll have `refined_labels.json`:

```json
{
  "0000.jpg": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "0001.jpg": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
}
```

This file is **ready for retraining** immediately!

## Step 3: Resume Later (Optional)

If you quit early, just run the same command again:

```powershell
python refine_corners.py --checkpoint outputs/heatmap_regression/best.pth
```

It automatically **skips already refined images** and continues where you left off.

## Step 4: Retrain with Refined Data

Once you've refined all images:

```powershell
# Retrain using refined labels
python src/train_heatmap.py `
  --config configs/heatmap_base.yaml `
  --labels_path refined_labels.json `
  --images_dir images `
  --output_dir outputs/heatmap_v2
```

**Expected improvement:**
- Original model: 5.79px MAE
- After refinement: 3-4px MAE (30-40% better)
- IoU: 0.93 → 0.96+

## Advanced Usage

### Auto-Generate Predictions (No GUI)

For large batches, generate predictions first:

```powershell
python refine_corners.py --checkpoint outputs/heatmap_regression/best.pth --auto
```

Then review/fix manually later.

### Export Model to ONNX

For deployment (mobile, web, edge devices):

```powershell
python refine_corners.py `
  --checkpoint outputs/heatmap_regression/best.pth `
  --export-onnx exports/heatmap_model.onnx
```

Creates portable ONNX file alongside your refinement.

### Process Specific Images Only

Move images to refine into a separate folder:

```powershell
mkdir new_images
# Copy your new images here
python refine_corners.py `
  --checkpoint outputs/heatmap_regression/best.pth `
  --images-dir new_images `
  --output new_labels.json
```

## Control Reference Card

```
┌─────────────────────────────────────────────────┐
│         Interactive Controls                     │
├─────────────────────────────────────────────────┤
│ Mouse:                                           │
│   Left-click    → Select nearest corner          │
│   Drag          → Move selected corner           │
│   Release       → Apply position                 │
│                                                  │
│ Navigation:                                      │
│   n             → Save & Next image              │
│   b             → Back to previous               │
│   s             → Skip without saving            │
│                                                  │
│ Editing:                                         │
│   r             → Reset to model prediction      │
│   u             → Undo last corner move          │
│                                                  │
│ Progress:                                        │
│   w             → Write checkpoint (resume later)│
│   q             → Quit & save all                │
│   ESC           → Quit without saving current    │
└─────────────────────────────────────────────────┘
```

## Tips & Best Practices

### When to Adjust Corners:

✅ **DO adjust when:**
- Corner is clearly off (>5px from true corner)
- Document edge is obvious but prediction missed
- Model confused by shadow/lighting
- Perspective causes slight misalignment

❌ **DON'T adjust when:**
- Prediction is within 2-3px of true corner (model is good enough)
- You're unsure where true corner is (trust model)
- Adjustment would make quadrilateral non-convex

### Efficient Workflow:

1. **Quick scan first**: Press `n` through all images to see which need fixing
2. **Mark for review**: Note image numbers with bad predictions
3. **Second pass**: Press `b` to go back and fix marked images
4. **Checkpoint often**: Press `w` every 10-20 images

### Common Issues:

**Corners in wrong order?**
- Model predicts canonical order: TL, TR, BR, BL (clockwise from top-left)
- If your ground truth has different order, reorder them consistently

**Can't click corner?**
- Zoom into window (resize with mouse)
- Move window to avoid taskbar overlap
- Threshold is 20px - click closer to center

**Window too small/large?**
- Script auto-scales to 1200×900 max
- Manually resize OpenCV window as needed

## Integration with Existing Workflow

### Scenario 1: New Dataset
```
1. Collect images → images/
2. Run refinement → refined_labels.json
3. Train from scratch → outputs/heatmap_v1
```

### Scenario 2: Augment Existing
```
1. Add new images to images/
2. Run refinement (skips existing) → refined_labels.json
3. Merge with labels.json (manual or script)
4. Retrain → outputs/heatmap_v2
```

### Scenario 3: Fix Bad Labels
```
1. Load existing labels.json
2. Run refinement → overwrites bad corners
3. Outputs corrected refined_labels.json
4. Retrain with clean data
```

## Next Steps

Once refined:

1. **Compare models**:
   ```powershell
   python analyze_heatmap_model.py --model-path outputs/heatmap_v2/best.pth
   ```

2. **Visualize improvements**:
   - Check epoch visualizations in `outputs/heatmap_v2/visualizations/`
   - Compare MAE/IoU metrics

3. **Deploy best model**:
   ```powershell
   python src/export.py `
     --checkpoint outputs/heatmap_v2/best.pth `
     --model_type heatmap `
     --output_dir exports/production
   ```

---

**Ready to start!** Just run:
```powershell
python refine_corners.py --checkpoint outputs/heatmap_regression/best.pth
```

Good luck! 🚀

# NOTE (legacy): this tool writes PIXEL coordinates in the pre-2026 label
# format. The current dataset contract is normalized [0, 1] corners, which
# `refine_corners.py` produces. Kept for reference only.
import cv2
import json
import os

# ---------------- CONFIG ---------------- #
IMAGES_DIR = r"D:\Side Projects\Projects\PaperMask\dataset"
LABELS_JSON = "labels.json"
OUTPUT_DIR = r"D:\Side Projects\Projects\PaperMask\labeled_validation"
POINT_RADIUS = 8            # bigger points
POINT_COLOR = (0, 0, 255)   # Red
LINE_COLOR = (255, 0, 0)    # Blue rectangle
LINE_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)    # Green
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2
# ---------------------------------------- #

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load JSON labels
with open(LABELS_JSON, "r") as f:
    labels = json.load(f)

for filename, pts in labels.items():
    img_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(img_path):
        print(f"Image not found: {filename}")
        continue

    img = cv2.imread(img_path)

    # Draw points and point numbers
    for idx, pt in enumerate(pts):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), POINT_RADIUS, POINT_COLOR, -1)
        cv2.putText(img, str(idx+1), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

    # Draw lines connecting corners (rectangle)
    for i in range(4):
        pt1 = (int(pts[i][0]), int(pts[i][1]))
        pt2 = (int(pts[(i + 1) % 4][0]), int(pts[(i + 1) % 4][1]))
        cv2.line(img, pt1, pt2, LINE_COLOR, LINE_THICKNESS)

    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, img)

print(f"Validation images saved to: {OUTPUT_DIR}")

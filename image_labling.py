import cv2
import json
import os
import glob

# ---------------- CONFIG ---------------- #
IMAGES_DIR = r"D:\Side Projects\Projects\PaperMask\images"
OUTPUT_JSON = "labels.json"
IMAGE_EXT = "*.jpg"
POINTS_PER_IMAGE = 4
MAX_DISPLAY_DIM = 1000
ZOOM_STEP = 0.1
MIN_SCALE = 0.2
MAX_SCALE = 5.0
# ---------------------------------------- #

image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, IMAGE_EXT)))
labels = {}

if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r") as f:
        labels = json.load(f)

points = []
current_filename = ""
scale = 1.0
offset_x, offset_y = 0, 0
mouse_x, mouse_y = 0, 0

def mouse_callback(event, x, y, flags, param):
    global points, mouse_x, mouse_y
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < POINTS_PER_IMAGE:
            # Convert display coordinates to original image coordinates
            orig_x = int((x + offset_x) / scale)
            orig_y = int((y + offset_y) / scale)
            points.append([orig_x, orig_y])

def label_image(img_path):
    global points, current_filename, scale, offset_x, offset_y
    points = []
    offset_x, offset_y = 0, 0
    current_filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]

    # Initial scale to fit max dimension
    scale = min(1.0, MAX_DISPLAY_DIM / max(orig_h, orig_w))
    
    window_name = "Labeling"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        # Compute display window size and crop region
        disp_w = int(orig_w * scale)
        disp_h = int(orig_h * scale)
        display = cv2.resize(img, (disp_w, disp_h))

        # Apply offsets for panning
        canvas = display.copy()
        temp = canvas

        # Draw points
        for idx, pt in enumerate(points):
            pt_draw = (int(pt[0]*scale - offset_x), int(pt[1]*scale - offset_y))
            cv2.circle(temp, pt_draw, 5, (0, 0, 255), -1)
            cv2.putText(temp, str(idx+1), pt_draw, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.imshow(window_name, temp)
        key = cv2.waitKey(1)

        # ESC to skip
        if key == 27:
            print(f"Skipped {current_filename}")
            break
        # Zoom in/out with +/-
        elif key == ord('+') or key == ord('='):
            scale = min(scale + ZOOM_STEP, MAX_SCALE)
        elif key == ord('-') or key == ord('_'):
            scale = max(scale - ZOOM_STEP, MIN_SCALE)
        # Save points when all clicked
        if len(points) == POINTS_PER_IMAGE:
            labels[current_filename] = points.copy()
            print(f"Saved {POINTS_PER_IMAGE} points for {current_filename}")
            break

    cv2.destroyAllWindows()

for idx in range(len(image_files)):
    label_image(image_files[idx])

with open(OUTPUT_JSON, "w") as f:
    json.dump(labels, f, indent=2)

print("Labeling complete.")

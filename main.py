import cv2
import numpy as np
import math
import csv
import os

# ---------- PARAMETERS ----------
IMAGE_PATH = "./renders/whole.jpg"
OUTPUT_IMAGE = "dipoles_detected.jpg"
OUTPUT_CSV = "dipoles_detected.csv"

MIN_RADIUS = 5       # pixels (~0.35 m if 1 px ≈ 0.0716 m)
MAX_RADIUS = 10      # pixels (~0.7 m)
CIRCLE_SENSITIVITY = 20  # Hough param2, lower = more detections
CONTRAST_THRESHOLD = 25  # intensity difference between halves
GAUSSIAN_SIZE = (3,3)
CLAHE_CLIP = 2.0
CLAHE_TILE = (8,8)

# ---------- LOAD & PREPROCESS ----------
gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError(f"Could not read image at {IMAGE_PATH}")

# Normalize contrast and reduce noise
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
enhanced = clahe.apply(gray)
blurred = cv2.GaussianBlur(enhanced, GAUSSIAN_SIZE, 0)

# ---------- HOUGH CIRCLE DETECTION ----------
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=15,
    param1=100,
    param2=CIRCLE_SENSITIVITY,
    minRadius=MIN_RADIUS,
    maxRadius=MAX_RADIUS
)

output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
results = []

# ---------- ANALYZE EACH CIRCLE ----------
if circles is not None:
    circles = np.around(circles[0, :]).astype(int)
    for (x, y, r) in circles:
        # Clip ROI boundaries to image
        y1 = max(0, y - r)
        y2 = min(enhanced.shape[0], y + r)
        x1 = max(0, x - r)
        x2 = min(enhanced.shape[1], x + r)
        roi = enhanced[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            continue

        # Compute local gradients
        gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)

        # Weighted mean orientation (dipole axis)
        avg_angle = np.arctan2(np.mean(np.sin(angle) * magnitude),
                               np.mean(np.cos(angle) * magnitude))

        # Sample intensity along that axis
        line_profile = []
        for i in range(-r, r):
            dx = int(round(x + i * math.cos(avg_angle)))
            dy = int(round(y + i * math.sin(avg_angle)))
            if 0 <= dy < enhanced.shape[0] and 0 <= dx < enhanced.shape[1]:
                line_profile.append(enhanced[dy, dx])

        if len(line_profile) < 4:
            continue

        line_profile = np.array(line_profile)
        n = len(line_profile)//2
        contrast = abs(np.mean(line_profile[:n]) - np.mean(line_profile[n:]))

        # If strong dipolar contrast → mark detection
        if contrast > CONTRAST_THRESHOLD:
            cv2.circle(output, (x, y), r, (0, 255, 0), 1)
            endx = int(x + r * math.cos(avg_angle))
            endy = int(y + r * math.sin(avg_angle))
            cv2.arrowedLine(output, (x, y), (endx, endy), (0, 0, 255), 1, tipLength=0.3)

            # Save detection record
            results.append({
                "x_px": x,
                "y_px": y,
                "radius_px": r,
                "contrast": round(float(contrast), 2),
                "orientation_deg": round(math.degrees(avg_angle) % 180, 1)  # 0–180 range
            })

# ---------- OUTPUT RESULTS ----------
cv2.imwrite(OUTPUT_IMAGE, output)

if results:
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"✅ Detected {len(results)} dipoles.")
    print(f"Results saved to {os.path.abspath(OUTPUT_IMAGE)}")
    print(f"Data saved to {os.path.abspath(OUTPUT_CSV)}")
else:
    print("⚠️ No dipoles detected. Try lowering CONTRAST_THRESHOLD or adjusting Hough parameters.")

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_dipole_template(size, angle_deg):
    """
    Creates a balanced dipole kernel (sum = 0).
    """
    pad = int(size * 0.5)
    canvas_size = size + 2 * pad
    template = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    center = (canvas_size // 2, canvas_size // 2)
    radius = size // 2
    
    # Draw the dipole
    mask = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    cv2.circle(mask, center, radius, 1, -1)
    
    half_mask = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    cv2.rectangle(half_mask, (0, 0), (center[0], canvas_size), 1, -1)
    
    # We use +1 and -1 so the sum is roughly 0 (DC blocking)
    # This prevents the filter from just responding to bright areas
    template[mask == 1] = 1.0
    template[np.logical_and(mask == 1, half_mask == 1)] = -1.0
    
    # Rotate
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(template, M, (canvas_size, canvas_size))
    
    # Crop back
    y1, y2 = pad, pad + size
    x1, x2 = pad, pad + size
    return rotated[y1:y2, x1:x2]

def rotational_matched_filter(image, dipole_size, angular_step=15):
    """
    Computes the max response across rotations.
    """
    # 1. Preprocessing: Handle the 'No Data' black regions
    valid_mask = image > 10 
    
    # Calculate mean of valid pixels
    mean_val = np.mean(image[valid_mask])
    
    # --- FIX IS HERE ---
    # We explicitly cast the result back to float32 after subtraction
    img_float = (image.astype(np.float32) - mean_val).astype(np.float32)
    
    # Zero out the invalid regions again
    img_float[~valid_mask] = 0
    
    h, w = image.shape
    max_response_map = np.full((h, w), -np.inf, dtype=np.float32)
    best_angle_map = np.zeros((h, w), dtype=np.int16)
    
    # Determine padding needed for the template
    template_dummy = create_dipole_template(dipole_size, 0)
    th, tw = template_dummy.shape
    pad_h = (h - (h - th + 1)) // 2 
    pad_w = (w - (w - tw + 1)) // 2
    
    print(f"Scanning 0 to 360 deg in steps of {angular_step}...")
    
    for angle in np.arange(0, 360, angular_step):
        # Ensure kernel is also strictly float32
        kernel = create_dipole_template(dipole_size, angle).astype(np.float32)
        
        # Now both img_float and kernel are float32 -> No Error
        res = cv2.matchTemplate(img_float, kernel, cv2.TM_CCOEFF)
        
        res_h, res_w = res.shape
        full_res = cv2.copyMakeBorder(res, 
                                      pad_h, h - res_h - pad_h, 
                                      pad_w, w - res_w - pad_w,
                                      cv2.BORDER_CONSTANT, value=-np.inf)
        
        update_mask = full_res > max_response_map
        max_response_map[update_mask] = full_res[update_mask]
        best_angle_map[update_mask] = angle

    # Final cleanup
    max_response_map[~valid_mask] = -np.inf
    
    return max_response_map, best_angle_map

def vectorized_peak_detection(heatmap, percentile_threshold, min_distance):
    """
    Uses statistical percentile thresholding instead of MinMax normalization.
    """
    # 1. Statistical Thresholding
    # We only care about the top X% of matches.
    # Filter out -inf values for calculation
    valid_scores = heatmap[heatmap > -np.inf]
    
    if len(valid_scores) == 0:
        return [], heatmap
    
    # Calculate the score value at the Nth percentile (e.g., 99.9th percentile)
    thresh_val = np.percentile(valid_scores, percentile_threshold)
    print(f"Auto-calculated Threshold: {thresh_val:.2f} (Top {100-percentile_threshold:.2f}%)")
    
    # Create mask
    mask = heatmap > thresh_val
    
    # 2. Dilation (Local Max Finding)
    kernel_size = min_distance | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # We replace -inf with a very low number for dilation stability
    safe_heatmap = heatmap.copy()
    safe_heatmap[safe_heatmap == -np.inf] = np.min(valid_scores)
    
    dilated = cv2.dilate(safe_heatmap, kernel)
    
    # 3. Peak Determination
    is_peak = (safe_heatmap == dilated) & mask
    
    y_coords, x_coords = np.where(is_peak)
    detections = list(zip(x_coords, y_coords))
    
    return detections, safe_heatmap

# --- MAIN EXECUTION ---

# 1. Load your specific image
image_path = './renders/whole.jpg'  # Make sure this matches your file name
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Could not load {image_path}. Check filename.")
    exit()

# 2. Parameters (TUNED FOR YOUR IMAGE)
# Based on your image, the dipole with the red circle is roughly 20-30 pixels wide.
ARTIFACT_SIZE = 14      # Try 25. If too small, try 35.
ANGULAR_STEP = 15       # 15 degrees is fine
PERCENTILE = 99.00       # We want the top 0.2% of correlations. 
                        # Lower this to 99.5 if you miss some. Raise to 99.9 if too noisy.

# 3. Process
response_map, angle_map = rotational_matched_filter(image, ARTIFACT_SIZE, ANGULAR_STEP)

# 4. Detect
detections, final_heatmap = vectorized_peak_detection(response_map, PERCENTILE, min_distance=ARTIFACT_SIZE)

print(f"Detected {len(detections)} artifacts.")

# 5. Visual Output
output_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for (x, y) in detections:
    angle = angle_map[y, x]
    # Draw simple red circle
    cv2.circle(output_img, (x, y), ARTIFACT_SIZE//2 + 4, (0, 0, 255), 2)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Show the raw heatmap (critical for debugging)
# We crop the range so the massive outliers don't wash out the contrast
vmin, vmax = np.percentile(final_heatmap, [50, 99.9]) 
im1 = ax1.imshow(final_heatmap, cmap='jet', vmin=vmin, vmax=vmax)
ax1.set_title("Correlation Heatmap (Zoom in to see hotspots)")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

ax2.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
ax2.set_title(f"Detections (Top {100-PERCENTILE:.1f}%)\nUse Zoom Tool")
ax2.axis('off')

plt.tight_layout()
plt.show()
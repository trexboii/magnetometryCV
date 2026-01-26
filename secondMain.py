import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- HELPER FUNCTIONS ---

def create_dipole_template(size, angle_deg):
    """
    Creates a dipole template rotated by specific angle.
    Background is 0 (neutral) for TM_CCOEFF to work best with zero-padding,
    but we handle the logic to ensure we match your 'gray' background data.
    """
    # Create larger canvas to avoid cutting off corners during rotation
    pad = int(size * 0.5)
    canvas_size = size + 2 * pad
    
    # 1. Start with neutral gray (0 correlation weight)
    # We use float32 for precision during rotation
    template = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    
    center = (canvas_size // 2, canvas_size // 2)
    radius = size // 2
    
    # 2. Draw the Dipole Logic
    # We want: 
    # High Magnetic Field (Black in image) -> Negative Weight
    # Low Magnetic Field (White in image) -> Positive Weight
    # Neutral Background -> 0 Weight
    
    mask = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    cv2.circle(mask, center, radius, 1, -1)
    
    # Left half negative, Right half positive (before rotation)
    # This creates a "kernel" that responds to changes, ignoring flat averages.
    half_mask = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    cv2.rectangle(half_mask, (0, 0), (center[0], canvas_size), 1, -1)
    
    # Apply weights
    template[mask == 1] = 1.0  # White side (positive correlation)
    template[np.logical_and(mask == 1, half_mask == 1)] = -1.0 # Black side (negative)
    
    # 3. Rotate
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(template, M, (canvas_size, canvas_size))
    
    # 4. Crop back to center to keep kernel tight
    y1, y2 = pad, pad + size
    x1, x2 = pad, pad + size
    
    return rotated[y1:y2, x1:x2]

def rotational_matched_filter(image, dipole_size, angular_step=15):
    """
    Runs matched filter for multiple angles and returns the MAX response map
    and the angle map (which angle gave the best match).
    """
    # Pre-process image to align with our kernel weights
    # We center the image data around 0 so 'gray' background becomes 0.
    # Assuming your image is 0-255 uint8, and background is roughly 127.
    img_float = image.astype(np.float32) - 127.0
    
    h, w = image.shape
    
    # Initialize buffers for the max response
    # We start with negative infinity so any real correlation overwrites it
    max_response_map = np.full((h, w), -np.inf, dtype=np.float32)
    best_angle_map = np.zeros((h, w), dtype=np.int16)
    
    angles = np.arange(0, 360, angular_step)
    
    print(f"Scanning {len(angles)} angles...")
    
    for angle in angles:
        kernel = create_dipole_template(dipole_size, angle)
        
        # TM_CCOEFF: Matches pattern strength. 
        # Strong signal = High value. Weak signal (noise) = Low value.
        # No normalization means noise isn't artificially boosted.
        res = cv2.matchTemplate(img_float, kernel, cv2.TM_CCOEFF)
        
        # matchTemplate returns a result smaller than source by kernel size
        # We need to pad it back to match original image size for overlay
        pad_h = (h - res.shape[0]) // 2
        pad_w = (w - res.shape[1]) // 2
        
        # Create full-size response for this angle
        full_res = cv2.copyMakeBorder(res, pad_h, h-res.shape[0]-pad_h, 
                                      pad_w, w-res.shape[1]-pad_w, 
                                      cv2.BORDER_CONSTANT, value=-np.inf)
        
        # Update global maximums
        # Where this angle's response is greater than current max, update max and record angle
        update_mask = full_res > max_response_map
        max_response_map[update_mask] = full_res[update_mask]
        best_angle_map[update_mask] = angle

    return max_response_map, best_angle_map

def non_max_suppression_fast(heatmap, threshold, radius):
    """
    Locates peaks in the heatmap.
    """
    # Normalize heatmap to 0-1 range for easier thresholding by the user
    # Note: We do this AFTER finding the max correlation, preserving relative signal strength
    heatmap_norm = cv2.normalize(heatmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    coordinates = []
    temp_map = heatmap_norm.copy()
    
    # Optimization: ignore everything below threshold immediately
    temp_map[temp_map < threshold] = 0
    
    while True:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(temp_map)
        
        if max_val <= 0: # All peaks found
            break
            
        coordinates.append(max_loc)
        
        # "Erase" this peak so we can find the next one
        cv2.circle(temp_map, max_loc, radius, 0, -1)
        
    return coordinates

# --- MAIN SCRIPT ---

# 1. Load Data
# Replace this with: image = cv2.imread('your_file.png', cv2.IMREAD_GRAYSCALE)
# Generating synthetic data with strong and weak dipoles to test "noise" rejection
image = cv2.imread("./renders/whole.jpg",cv2.IMREAD_GRAYSCALE)

# 2. Parameters
ARTIFACT_SIZE = 40      # Diameter in pixels
ANGULAR_STEP = 20       # Degrees (smaller = more precision, slower)
THRESHOLD = 0.80        # 0.0 to 1.0 (relative to the strongest signal found)

# 3. Process
print("Starting Rotational Matched Filter...")
response_map, angle_map = rotational_matched_filter(image, ARTIFACT_SIZE, ANGULAR_STEP)

# 4. Detect
detections = non_max_suppression_fast(response_map, THRESHOLD, radius=ARTIFACT_SIZE//2)

print(f"Detected {len(detections)} artifacts.")
print("X, Y Coordinates:")
for p in detections:
    print(p)

# 5. Visualization (Interactive)
output_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Draw detections
for (x, y) in detections:
    angle = angle_map[y, x]
    # Draw circle
    cv2.circle(output_img, (x, y), ARTIFACT_SIZE//2 + 5, (0, 0, 255), 2)
    # Draw crosshair
    cv2.drawMarker(output_img, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
    # Label with angle
    cv2.putText(output_img, f"{angle}deg", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# Set up Matplotlib for Zooming
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
ax.set_title(f"Detected Dipoles (Threshold: {THRESHOLD})\nUse the Magnifying Glass tool below to Zoom")
ax.axis('off')

print("\n--- INTERACTION ---")
print("A window has opened.")
print("1. Use the 'Magnifying Glass' icon in the toolbar to zoom in.")
print("2. Use the 'Floppy Disk' icon to save the zoomed view.")
plt.show()
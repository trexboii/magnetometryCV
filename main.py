import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, uniform_filter
from scipy.signal import fftconvolve
from skimage.feature import peak_local_max
from skimage import io, color, img_as_ubyte
from skimage.filters import gaussian, threshold_otsu
from skimage.filters.rank import equalize
from skimage.morphology import disk
from skimage.transform import resize

# ------------------------------
# Parameters
# ------------------------------
dipole_radius = 150       # radius of each dipole
kernel_size = 100         # size of the kernel (should contain full dipole)
n_orientations = 12      # number of rotations to check
scale = 0.3 #image resize scale
threshold_rel = 0.2 #threshold clip

# ------------------------------
# Helper: Create black/white half-circle dipole kernel
# ------------------------------
print("creating dipole kernel...")
dipole_radius = int(dipole_radius * scale)
kernel_size = int(kernel_size * scale)
def create_half_circle_dipole_kernel(size, radius):
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, y)
    
    disk_mask = (X**2 + Y**2 <= radius**2).astype(float)
    disk_mask[X >= 0] *= -1  # left half +1, right half -1
    
    disk_mask -= disk_mask.mean()
    disk_mask /= np.std(disk_mask)
    return disk_mask

dipole_kernel = create_half_circle_dipole_kernel(kernel_size, dipole_radius)

# ------------------------------
# Load and preprocess real image
# ------------------------------
image_path = r"C:\Users\ASUS ZenBook 14\Desktop\Python Projects\magnetometryCV\renders\whole.jpg"  # ← change this path!


print("filtering image...")
image_rgb = io.imread(image_path)
if image_rgb.ndim == 3:
    image = color.rgb2gray(image_rgb)
else:
    image = image_rgb.astype(float)
image = resize(image, (int(image.shape[0]*scale), int(image.shape[1]*scale)),
               anti_aliasing=True)
# Light smoothing + normalization
image = gaussian(image, sigma=1)
image = (image - image.mean()) / image.std()

# Local contrast enhancement 
image_ubyte = img_as_ubyte((image - image.min()) / (image.max() - image.min()))
image_eq = equalize(image_ubyte, footprint=disk(dipole_radius))
image = image_eq / 255.0
image = (image - image.mean()) / image.std()

# ------------------------------
# Matched filtering across rotations
# ------------------------------
response = np.zeros_like(image)
print("steerable matched filtering...")
for i in range(n_orientations):
    angle = i * 360 / n_orientations
    rotated_kernel = rotate(dipole_kernel, angle, reshape=False)
    conv = fftconvolve(image, rotated_kernel, mode='same')
    
    # Local normalization (reduces false edges)
    local_mean = uniform_filter(image, size=dipole_radius)
    local_sq_mean = uniform_filter(image**2, size=dipole_radius)
    local_var = np.clip(local_sq_mean - local_mean**2, a_min=0, a_max=None)
    local_std = np.sqrt(local_var)
    conv_norm = conv - local_std  # less harsh normalization


    # Clean up any numerical issues
    conv_norm[~np.isfinite(conv_norm)] = 0

    
    response = np.maximum(response, conv_norm)

# ------------------------------
# Mask out image borders
# ------------------------------

border = dipole_radius
response[:border, :] = 0
response[-border:, :] = 0
response[:, :border] = 0
response[:, -border:] = 0

# ------------------------------
# Adaptive thresholding and peak detection
# ------------------------------
print("detecting dipoles...")

peaks = peak_local_max(response, min_distance=dipole_radius, threshold_rel=threshold_rel)


# ------------------------------
# Post-filter peaks by local contrast symmetry
# ------------------------------
valid_peaks = []
for y, x in peaks:
    patch = image[max(0, y - dipole_radius):y + dipole_radius,
                  max(0, x - dipole_radius):x + dipole_radius]
    if patch.shape[0] < dipole_radius or patch.shape[1] < dipole_radius:
        continue
    left_mean = np.mean(patch[:, :patch.shape[1]//2])
    right_mean = np.mean(patch[:, patch.shape[1]//2:])
    contrast = abs(left_mean - right_mean)
    if contrast > 0.2:  # tweak this threshold if needed
        valid_peaks.append((y, x))
peaks = np.array(valid_peaks)

# ------------------------------
# Visualization
# ------------------------------
print("creating figures...")
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(image, cmap='gray')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Matched Filter Response")
plt.imshow(response, cmap='hot')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Detected Dipoles (Filtered)")
plt.imshow(image, cmap='gray')
if len(peaks) > 0:
    plt.scatter(peaks[:, 1], peaks[:, 0], color='lime', s=60, marker='x')
plt.colorbar()

plt.tight_layout()
plt.show()

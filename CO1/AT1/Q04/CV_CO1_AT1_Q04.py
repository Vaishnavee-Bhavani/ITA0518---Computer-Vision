import cv2 # Import OpenCV for image processing algorithms
import numpy as np # Import Numpy for matrix operations
import matplotlib.pyplot as plt # Import Matplotlib to render plots

# Step 1: Create a synthetic medical phantom (representing a brain scan slice)
phantom = np.zeros((256, 256), dtype=np.uint8)

# Simulate skull structure (outer boundary)
cv2.ellipse(phantom, (128, 128), (100, 120), 0, 0, 360, 80, -1)

# Simulate general brain matter
cv2.ellipse(phantom, (128, 128), (90, 110), 0, 0, 360, 140, -1)

# Simulate subtle internal soft tissues (ventricles with near-similar gray levels)
cv2.ellipse(phantom, (105, 120), (25, 45), 15, 0, 360, 165, -1)
cv2.ellipse(phantom, (151, 120), (25, 45), -15, 0, 360, 165, -1)

# Simulate a tiny pathology/microcalcification (high spatial detail, requires high pixel res)
cv2.circle(phantom, (128, 70), 3, 255, -1) 

# Step 2: Simulate Low Pixel (Spatial) Resolution
# Downsample to a tiny 24x24 grid, then scale back up to show the structural loss
low_spatial_raw = cv2.resize(phantom, (24, 24), interpolation=cv2.INTER_AREA)
low_spatial = cv2.resize(low_spatial_raw, (256, 256), interpolation=cv2.INTER_NEAREST)

# Step 3: Simulate Low Intensity (Quantization) Resolution
# Compress the original 256 gray levels down to just 4 rigid levels (2-bit depth)
low_intensity = (np.floor(phantom / 64) * 64).astype(np.uint8)

# Step 4: Display the diagnostic impact side-by-side
plt.figure(figsize=(12, 5))

# Original Reference
plt.subplot(131)
plt.imshow(phantom, cmap='bone')
plt.title('High Res (Reference)')
plt.axis('off')

# Spatial Resolution Degradation
plt.subplot(132)
plt.imshow(low_spatial, cmap='bone')
plt.title('Low Pixel Res\n(Tiny Tumor Erased)')
plt.axis('off')

# Intensity Resolution Degradation
plt.subplot(133)
plt.imshow(low_intensity, cmap='bone')
plt.title('Low Intensity Res\n(Tissue Contrast Lost)')
plt.axis('off')

plt.tight_layout()
plt.show() # Show the final plot window to the user

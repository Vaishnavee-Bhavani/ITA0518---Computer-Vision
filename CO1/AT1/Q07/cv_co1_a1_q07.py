import cv2                                                  # Import OpenCV for image processing algorithms
import numpy as np                                          # Import Numpy for mathematical matrix operations
import matplotlib.pyplot as plt                             # Import Matplotlib to visualize the output on screen

# =====================================================================
# STEP 1: CREATE A PERFECTLY SMOOTH GRAYSCALE GRADIENT (Original Scene)
# =====================================================================
gradient_line = np.arange(0, 256, dtype=np.uint8)
original_smooth = np.tile(gradient_line, (256, 1))

# =====================================================================
# STEP 2: APPLY QUANTIZATION (Simulating Low Digital Bit-Depth)
# =====================================================================
shades_target = 4
step_size = 256 // shades_target
quantized_low = (original_smooth // step_size) * step_size

# =====================================================================
# STEP 3: VISUALIZE IN A COMPACT SIZE (figsize altered to 5x2.5)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(5, 2.5)) 

# Plot 1: High Bit-Depth (Smooth Detail)
axes[0].imshow(original_smooth, cmap='gray')
axes[0].set_title('8-Bit (256 Shades)', fontsize=8)
axes[0].axis('off')

# Plot 2: Low Bit-Depth (Banded Detail)
axes[1].imshow(quantized_low, cmap='gray')
axes[1].set_title('2-Bit (4 Shades)', fontsize=8)
axes[1].axis('off')

plt.tight_layout()
plt.show()

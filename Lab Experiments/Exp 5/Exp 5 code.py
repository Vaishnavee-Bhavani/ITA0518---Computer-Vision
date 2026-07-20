import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow

# -------------------------------------------------------------
# Step 1: Upload an image from your computer
# -------------------------------------------------------------
print("Please upload an image file:")
uploaded = files.upload()

# Get the uploaded image path automatically
image_path = list(uploaded.keys())[0]

# -------------------------------------------------------------
# Step 2: Read the image using OpenCV
# -------------------------------------------------------------
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read image!")
else:
    # -------------------------------------------------------------
    # Step 3: Define a structuring element (kernel) and Erode
    # -------------------------------------------------------------
    # Create a 5x5 matrix of ones as the structuring element
    kernel = np.ones((5, 5), np.uint8)

    # Perform Erosion
    eroded_image = cv2.erode(image, kernel, iterations=1)

    # -------------------------------------------------------------
    # Step 4: Display the results in Colab
    # -------------------------------------------------------------
    print("\n--- Original Image ---")
    cv2_imshow(image)

    print("\n--- Eroded Image ---")
    cv2_imshow(eroded_image)

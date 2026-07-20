import cv2
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
    # Step 3: Apply Canny Edge Detection
    # -------------------------------------------------------------
    # Syntax: cv2.Canny(image, threshold1, threshold2)
    # threshold1 (100): Lower hysteresis threshold
    # threshold2 (200): Upper hysteresis threshold
    edges = cv2.Canny(image, 100, 200)

    # -------------------------------------------------------------
    # Step 4: Display the results in Colab
    # -------------------------------------------------------------
    print("\n--- Original Image ---")
    cv2_imshow(image)

    print("\n--- Canny Edge / Outline Image ---")
    cv2_imshow(edges)

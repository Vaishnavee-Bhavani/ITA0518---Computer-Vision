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
    # Step 3: Apply GaussianBlur
    # -------------------------------------------------------------
    # (15, 15) is the kernel size (must be odd positive integers)
    # 0 sets the standard deviation (sigmaX) to be computed automatically from kernel size
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

    # -------------------------------------------------------------
    # Step 4: Display the results in Colab
    # -------------------------------------------------------------
    print("\n--- Original Image ---")
    cv2_imshow(image)

    print("\n--- Gaussian Blurred Image ---")
    cv2_imshow(blurred_image)

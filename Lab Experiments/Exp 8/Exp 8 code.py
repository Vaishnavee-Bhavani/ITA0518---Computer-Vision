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
    # Get original dimensions
    height, width = image.shape[:2]
    print(f"Original Dimensions: {width}x{height}")

    # -------------------------------------------------------------
    # Step 3: Scale to Bigger and Smaller sizes
    # -------------------------------------------------------------
    # Option A: Scaling using scale factors (fx, fy)
    bigger_image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    smaller_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # Option B: Alternatively, scale using absolute pixel dimensions:
    # bigger_image  = cv2.resize(image, (int(width * 1.5), int(height * 1.5)))
    # smaller_image = cv2.resize(image, (int(width * 0.5), int(height * 0.5)))

    # -------------------------------------------------------------
    # Step 4: Display the results in Colab
    # -------------------------------------------------------------
    print(f"\n--- Original Image ({width}x{height}) ---")
    cv2_imshow(image)

    print(f"\n--- Bigger Image (1.5x) ({bigger_image.shape[1]}x{bigger_image.shape[0]}) ---")
    cv2_imshow(bigger_image)

    print(f"\n--- Smaller Image (0.5x) ({smaller_image.shape[1]}x{smaller_image.shape[0]}) ---")
    cv2_imshow(smaller_image)

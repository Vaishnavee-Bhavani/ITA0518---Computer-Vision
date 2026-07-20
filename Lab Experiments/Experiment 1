import cv2
from google.colab import files
from google.colab.patches import cv2_imshow

# 1. Prompt to upload an image from your computer
uploaded = files.upload()

# Get the name of the uploaded file
image_path = list(uploaded.keys())[0]

# 2. Read the image
image = cv2.imread(image_path)

# 3. Convert to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 4. Display results
print("\n--- Original Image ---")
cv2_imshow(image)

print("\n--- Grayscale Image ---")
cv2_imshow(gray_image)

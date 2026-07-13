
import cv2                                     # Import OpenCV library for image processing
import matplotlib.pyplot as plt                # Import Matplotlib for displaying images

# Read the image from the disk into a NumPy array
img = cv2.imread('number_plate.jpg')           # cv2.imread reads the image in BGR (Blue-Green-Red) format
# Convert the BGR image to RGB format since Matplotlib expects RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # This ensures colors look correct when displayed

# Display the image using Matplotlib
plt.imshow(img_rgb)                            # Render the RGB image array
plt.title('Original Image')                    # Add a title to the plot window
plt.axis('off')                                # Hide the X and Y axes (ticks and borders) for a clean look
plt.show()                                     # Show the actual window on screen

# Print the dimensions of the loaded image
print("Image Shape:", img.shape)               # Outputs (height, width, channels) e.g., (100, 400, 3)

# Downsample the image to a tiny 50x25 resolution to simulate a very low sampling rate (bad camera sensor)
low_res = cv2.resize(img, (50, 25), interpolation=cv2.INTER_NEAREST) # INTER_NEAREST simply drops pixels without blending
print("Low-Res Shape:", low_res.shape)         # The image is now only 25 pixels high and 50 pixels wide

# Scale the tiny image back up to 400x200 so we can actually see the blocky pixelation effect on screen
pixelated = cv2.resize(low_res, (400, 200), interpolation=cv2.INTER_NEAREST) # Upscale using Nearest Neighbor to keep it blocky

# Create a figure with a specific size (width=10, height=4 inches)
plt.figure(figsize=(10, 4))                                            # Create a visualization window to hold our images

# Create the left subplot (1 row, 2 columns, 1st position)
plt.subplot(1, 2, 1)                                                   # Position this image within a grid layout
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Display the original, high-quality image
plt.title('Original (High Sampling)')                                  # Set a descriptive title for this plot
plt.axis('off')                                                        # Hide axis ticks and coordinates

# Create the right subplot (1 row, 2 columns, 2nd position)
plt.subplot(1, 2, 2)                                                   # Position this image within a grid layout
plt.imshow(cv2.cvtColor(pixelated, cv2.COLOR_BGR2RGB)) # Display the blurred/blocky low-sampling image
plt.title('Pixelated (Low Sampling)')                                  # Set a descriptive title for this plot
plt.axis('off')                                                        # Hide axis ticks and coordinates
plt.show()                                     # Render both subplots to the screen

# Define a list of different resolutions to test (Low, Medium, High)
sizes = [(50, 25), (200, 100), (400, 200)]     # Width x Height tuples
# Labels for our plot titles
labels = ['Low (50x25)', 'Medium (200x100)', 'High (400x200)']         # Define corresponding text labels

plt.figure(figsize=(14, 4))                    # Create a wide figure window

# Loop through each size and label simultaneously
for i, (size, label) in enumerate(zip(sizes, labels)):                 # Iterate through the parameters
    # Step 1: Downsample the original image to the specific 'size' to simulate the sensor resolution
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA) # INTER_AREA is best for shrinking images

    # Step 2: Upsample it back to 400x200 so we can compare them all at the same physical size
    restored = cv2.resize(small, (400, 200), interpolation=cv2.INTER_NEAREST)  # Upsample image

    plt.subplot(1, 3, i + 1)                   # Create a subplot (1 row, 3 columns, position i+1)
    plt.imshow(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)) # Show the image
    plt.title(label)                           # Set the title to the current sampling rate label
    plt.axis('off')                            # Hide the axis

plt.suptitle('Effect of Different Sampling Rates') # Add a main title for the whole window
plt.tight_layout()                             # Automatically adjust spacing between plots
plt.show()                                     # Display the final comparison

# PSNR = Peak Signal-to-Noise Ratio. It mathematically compares a degraded image to the original. Higher dB is better!
original = cv2.resize(img, (400, 200))         # Set our baseline ground truth image

# Test 3 different sampling sizes
for label, size in zip(['Low', 'Medium', 'High'], [(50,25), (200,100), (400,200)]):  # Iterate through the parameters
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)             # Simulate sampling
    restored = cv2.resize(small, (400, 200), interpolation=cv2.INTER_CUBIC) # Correct using Bicubic interpolation

    # Calculate the mathematical quality score between the original and the restored image
    psnr = cv2.PSNR(original, restored)                                # Compute Peak Signal-to-Noise Ratio

    print(f"{label} Sampling -> PSNR: {psnr:.2f} dB") # Print the result (e.g., "Low Sampling -> PSNR: 15.2 dB")

# Start with a heavily downsampled (blurred) low-resolution image
low_res = cv2.resize(img, (50, 25), interpolation=cv2.INTER_AREA)      # Apply resize downsampling

# Define a dictionary of the 4 mathematical interpolation methods OpenCV provides for upscaling
methods = {                                                            # Map names to OpenCV interpolation flags
    'Nearest Neighbor': cv2.INTER_NEAREST,     # Fast, but creates blocky pixelated edges
    'Bilinear': cv2.INTER_LINEAR,              # Uses 4 nearby pixels, creates smooth but slightly blurry edges
    'Bicubic': cv2.INTER_CUBIC,                # Uses 16 nearby pixels, creates sharp and smooth edges (Best balance)
    'Lanczos': cv2.INTER_LANCZOS4              # Uses math formulas over 8x8 pixels, highest quality but slow
}

plt.figure(figsize=(14, 4))                    # Create a wide figure to fit 4 images

# Loop through our dictionary of methods
for i, (name, method) in enumerate(methods.items()):                   # Iterate through the parameters
    # Here is the correction! Upscale the low-res image using the current mathematical method
    upscaled = cv2.resize(low_res, (400, 200), interpolation=method)

    plt.subplot(1, 4, i + 1)                   # Place image in a 1x4 grid
    plt.imshow(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB and show
    plt.title(name, fontsize=9)                # Label the image with the method name
    plt.axis('off')                                                    # Hide axis ticks and coordinates

plt.suptitle('Interpolation Methods Comparison')                       # Set a master title for the entire window
plt.tight_layout()                                                     # Automatically fix padding and spacing
plt.show()                                     # Display the corrected images side-by-side

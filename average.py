import os
import cv2
import numpy as np

# Path to the folder containing generated saliency maps
saliency_maps_path = "E:/Digital Image Processing/Assignment 3/TranSalNet-Res/Datasets/Saliency_map_TD"

# Get list of all saliency map files (assuming they are saved as .jpg or .png)
saliency_files = [f for f in os.listdir(saliency_maps_path) if f.endswith(('.jpg', '.png'))]
saliency_files.sort()  # Ensure proper order

# Initialize an accumulator for sum of all saliency maps
average_map = None
count = len(saliency_files)

# Find the dimensions of the first image to resize others
first_img = cv2.imread(os.path.join(saliency_maps_path, saliency_files[0]), cv2.IMREAD_GRAYSCALE)
height, width = first_img.shape

for i, file in enumerate(saliency_files):
    img = cv2.imread(os.path.join(saliency_maps_path, file), cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    img = cv2.resize(img, (width, height))  # Resize to match the first image
    img = img.astype(np.float32)  # Convert to float for averaging

    if average_map is None:
        average_map = np.zeros_like(img, dtype=np.float32)

    average_map += img  # Sum pixel values

# Compute the final average
average_map /= count

# Normalize to 0-255 range for visualization (optional)
average_map = cv2.normalize(average_map, None, 0, 255, cv2.NORM_MINMAX)

# Save the averaged saliency map
output_path = os.path.join(saliency_maps_path, "average_saliency_map.jpg")
cv2.imwrite(output_path, average_map.astype(np.uint8))

print(f"Average saliency map saved at {output_path}")

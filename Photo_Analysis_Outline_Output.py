import cv2
import numpy as np
import os
from PIL import Image
import pandas as pd

# Define the threshold for binary conversion (0-255)
threshold = 92

# Function to calculate black pixel ratio within the polygon and save the image with a red outline
def process_image(image_path, output_folder):
    # Open the image using PIL and convert it to grayscale
    image = Image.open(image_path)
    grayscale_image = image.convert('L')

    # Apply a threshold to convert the grayscale image to black and white
    binary_image = grayscale_image.point(lambda p: 255 if p > threshold else 0)

    # Convert the image to a NumPy array for OpenCV processing
    binary_np = np.array(binary_image)

    # Convert to grayscale and apply binary threshold
    _, binary_cv = cv2.threshold(binary_np, threshold, 255, cv2.THRESH_BINARY)

    # Apply morphological opening and closing to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Adjust kernel size as needed
    morphed = cv2.morphologyEx(binary_cv, cv2.MORPH_OPEN, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)

    # Apply Canny edge detection to detect edges
    edges = cv2.Canny(morphed, 50, 150)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area (assuming it's the polygon)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw the red outline around the largest contour
        output_image = cv2.cvtColor(binary_np, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(output_image, [largest_contour], -1, (0, 0, 255), 2)  # Red color in BGR

        # Save the output image
        output_image_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_image_path, output_image)

        # Create a mask of the same size as the image, initialized to black (0)
        mask = np.zeros_like(binary_cv)

        # Fill the largest contour on the mask with white (255)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Get the number of pixels inside the polygon (white in the mask)
        total_polygon_pixels = np.sum(mask == 255)

        # Count the number of black pixels (0 in the binary image) within the polygon area
        black_pixels_within_polygon = np.sum((binary_cv == 0) & (mask == 255))

        # Calculate the ratio of black pixels within the polygon
        if total_polygon_pixels > 0:
            black_pixel_ratio = black_pixels_within_polygon / total_polygon_pixels
        else:
            black_pixel_ratio = 0  # Avoid division by zero if the contour is invalid

        return black_pixel_ratio
    else:
        # Return None if no polygon was found
        return None

# Directory containing the original images
input_folder = r'D:\Documenten\University\MS2\CADCAM 3 - Research\Photos'
output_folder = r'D:\Documenten\University\MS2\CADCAM 3 - Research\Photos\Converted'


# List to store the results
results = []

# Process each image in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(input_folder, filename)
        black_pixel_ratio = process_image(input_path, output_folder)
        if black_pixel_ratio is not None:
            results.append({"Image": filename, "Black Pixel Ratio": black_pixel_ratio})

# Convert the results to a DataFrame and save to a CSV file
df = pd.DataFrame(results)
output_csv_path = os.path.join(output_folder, 'black_pixel_ratios.csv')
df.to_csv(output_csv_path, index=False)

print(f"Images saved to {output_folder}. Black pixel ratio calculation completed! Results saved to {output_csv_path}.")

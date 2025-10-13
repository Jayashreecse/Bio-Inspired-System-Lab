# Step 1: Upload image from your local machine
from google.colab import files
uploaded = files.upload()  # Choose your noisy image file here

# Get the filename of the uploaded image
image_path = list(uploaded.keys())[0]
print(f"Uploaded file: {image_path}")

# Step 2: Import libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Step 3: Define median filter function
def median_filter(image_array, iterations=1):
    filtered_image = image_array.copy()
    height, width = filtered_image.shape
    
    for _ in range(iterations):
        new_image = filtered_image.copy()
        for i in range(height):
            for j in range(width):
                neighbors = []
                for x in range(max(0, i-1), min(height, i+2)):
                    for y in range(max(0, j-1), min(width, j+2)):
                        neighbors.append(filtered_image[x, y])
                neighbors.sort()
                median_value = neighbors[len(neighbors)//2]
                new_image[i, j] = median_value
        filtered_image = new_image
        
    return filtered_image

# Step 4: Load image as grayscale numpy array
def load_image(path):
    img = Image.open(path).convert('L')  # Grayscale
    return np.array(img)

# Step 5: Display original and filtered images side-by-side
def show_images(original, filtered):
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Filtered Image")
    plt.imshow(filtered, cmap='gray')
    plt.axis('off')
    plt.show()

# Step 6: Run the filtering process
noisy_image = load_image(image_path)
filtered_image = median_filter(noisy_image, iterations=3)
show_images(noisy_image, filtered_image)

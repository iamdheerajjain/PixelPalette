import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_dominant_colors(image_path, k=5):
    # Load image using OpenCV and convert BGR to RGB
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape image to 2D array of pixels
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixel_values)

    # Get the cluster centers (dominant colors)
    colors = np.uint8(kmeans.cluster_centers_)

    # Count the number of pixels in each cluster
    labels = kmeans.labels_
    counts = np.bincount(labels)

    # Sort colors by size
    sorted_idx = np.argsort(-counts)
    sorted_colors = colors[sorted_idx]
    sorted_counts = counts[sorted_idx]

    # Show the image and color bar
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    # Plot the color palette
    plt.subplot(1, 2, 2)
    plt.title(f"{k} Dominant Colors")
    plt.axis('off')
    bar = np.zeros((50, 300, 3), dtype='uint8')
    start_x = 0
    for idx, color in enumerate(sorted_colors):
        end_x = start_x + int((sorted_counts[idx] / sum(sorted_counts)) * 300)
        bar[:, start_x:end_x, :] = color
        start_x = end_x
    plt.imshow(bar)
    plt.show()

    return sorted_colors

# Example usage
image_path = '1.png'  # Replace with your image path
dominant_colors = find_dominant_colors(image_path, k=5)
print("Dominant Colors (RGB):")
print(dominant_colors)

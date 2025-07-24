import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_dominant_colors(image_path, k=5):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixel_values)
    colors = np.uint8(kmeans.cluster_centers_)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    sorted_idx = np.argsort(-counts)
    sorted_colors = colors[sorted_idx]
    sorted_counts = counts[sorted_idx]
    
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
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

image_path = '1.png'
dominant_colors = find_dominant_colors(image_path, k=5)
print("Dominant Colors (RGB):")
print(dominant_colors)

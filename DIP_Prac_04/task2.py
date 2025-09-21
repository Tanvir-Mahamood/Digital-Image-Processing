import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_filter(image, mask, nonlinear=False):
    m, n = mask.shape
    pad_h, pad_w = m // 2, n // 2
    padded = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    output = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+m, j:j+n]
            if nonlinear:
                output[i, j] = np.median(region)
            else:
                output[i, j] = np.sum(region * mask)
    return output

# Read grayscale image
img1 = cv2.imread("images/lenna.png", cv2.IMREAD_GRAYSCALE)

# Average filter (for unsharp masking)
avg_mask = np.ones((3,3), np.float32) / 9

# Step 1: Smooth image
avg_img = apply_filter(img1, avg_mask)

# Step 2: Compute mask (convert to float)
img1_f = img1.astype(np.float32)
mask = img1_f - avg_img

# Step 3: Unsharp masking (k = 1)
unsharp = img1_f + mask

# Step 4: High-boost filtering (k > 1)
k = 2.0
high_boost = img1_f + k * mask

# Convert back to uint8 safely
unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
high_boost = np.clip(high_boost, 0, 255).astype(np.uint8)
mask = np.clip(mask, 0, 255).astype(np.uint8)

# Show results
titles = ["Original", "Blurred (Avg)", "Mask", "Unsharp (k=1)", "High-boost (k=2)"]
images = [img1, avg_img.astype(np.uint8), mask, unsharp, high_boost]

plt.figure(figsize=(12, 6))
for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")
plt.tight_layout()
plt.show()

import cv2
from matplotlib import pyplot as plt

# Read image
img = cv2.imread("images/lenna.png")

# Convert from BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display with matplotlib
plt.imshow(img_rgb)
plt.title("Sample Image")
plt.axis("off")
plt.show()

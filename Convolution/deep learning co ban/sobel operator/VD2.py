import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

# Load the image in grayscale
image = cv2.imread('../Hough transform algorithm/1.png', cv2.IMREAD_GRAYSCALE)

# Define the horizontal and vertical Sobel kernels
sobel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
print(image)
# Apply the Sobel operator using scipy.ndimage.convolve
horizontal_gradient = ndimage.convolve(image, sobel_horizontal)
vertical_gradient = ndimage.convolve(image, sobel_vertical)
print(np.where(horizontal_gradient != 0))
print(horizontal_gradient[np.where(horizontal_gradient != 0)].shape)
print(vertical_gradient)
# Calculate the gradient magnitude
gradient_magnitude = np.sqrt(np.square(horizontal_gradient) + np.square(vertical_gradient)).astype(np.uint8)

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(horizontal_gradient, cmap='gray')
plt.title('Sobel Horizontal Gradient')

plt.subplot(2, 2, 3)
plt.imshow(vertical_gradient, cmap='gray')
plt.title('Sobel Vertical Gradient')

plt.subplot(2, 2, 4)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude')

plt.show()

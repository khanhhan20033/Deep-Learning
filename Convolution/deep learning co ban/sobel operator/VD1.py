import numpy as np
from scipy.ndimage import convolve
import scipy.ndimage
import imageio
import matplotlib.pyplot as plt

# Load the image
image = imageio.imread('1.png', as_gray=True)

# Define the Sobel kernels
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Apply the Sobel kernels to the image
gradient_x = convolve(image, sobel_x)
gradient_y = convolve(image, sobel_y)

# Calculate gradient magnitude and direction
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_direction = np.arctan2(gradient_y, gradient_x)

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(gradient_x, cmap='gray')
plt.title('Sobel Gradient (X-direction)')

plt.subplot(2, 2, 3)
plt.imshow(gradient_y, cmap='gray')
plt.title('Sobel Gradient (Y-direction)')

plt.subplot(2, 2, 4)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude')

plt.show()

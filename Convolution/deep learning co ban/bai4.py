'''
import cv2
import numpy as np


def sobel_edge_detection(image):
    # Chuyển ảnh thành ảnh xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tính gradient theo hướng ngang và dọc
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Áp dụng convolution với kernel Sobel để tính gradient
    gradient_x = cv2.filter2D(gray_image, -1, sobel_x)
    gradient_y = cv2.filter2D(gray_image, -1, sobel_y)

    # Tính toán độ lớn gradient
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Áp dụng ngưỡng để phát hiện biên cạnh
    threshold = 100
    edge_image = np.uint8(gradient_magnitude > threshold) * 255

    return edge_image


# Đọc ảnh đầu vào
image = cv2.imread('1.png')

# Áp dụng thuật toán Sobel để phát hiện biên cạnh
edge_image = sobel_edge_detection(image)

# Hiển thị ảnh kết quả
cv2.imshow('Sobel Edge Detection', edge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import cv2
import numpy as np

# Đọc ảnh đầu vào
import cv2
import numpy as np

# Load the image
image = cv2.imread('Hough transform algorithm/1.png', cv2.IMREAD_GRAYSCALE)

# Apply Sobel operator
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal gradient
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical gradient

# Calculate gradient magnitude and direction
magnitude = np.sqrt(sobelx**2 + sobely**2)
direction = np.arctan2(sobely, sobelx)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Gradient Magnitude', magnitude.astype(np.uint8))
cv2.imshow('Gradient Direction', direction)

cv2.waitKey(0)
cv2.destroyAllWindows()

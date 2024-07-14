import cv2
import numpy as np
# (a) Load và hiển thị ảnh màu/xám
image_color = cv2.imread('dangky.png')  # Thay 'image.jpg' bằng đường dẫn đến ảnh của bạn
cv2.imshow('Color Image', image_color)
cv2.waitKey(0)

# (b) Chuyển từ ảnh màu sang ảnh xám
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image', image_gray)
cv2.waitKey(0)

# (c) Resize ảnh
resized_image = cv2.resize(image_color, (800, 600))  # Thay new_width và new_height bằng kích thước mới
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)

# (d) Rotate ảnh
(h, w) = image_color.shape[:2]
center = (w // 2, h // 2)
angle = 45  # Góc xoay
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_image = cv2.warpAffine(image_color, M, (w, h))
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)

# (e) Threshold trên ảnh
_, threshold_image = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Image', threshold_image)
cv2.waitKey(0)
# Phát hiện cạnh bằng phương pháp Canny
edges = cv2.Canny(image_gray, 50, 150, apertureSize=3)

# Phát hiện đường thẳng
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Phát hiện đường tròn
circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                           param1=50, param2=30, minRadius=0, maxRadius=0)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(image_color, (x, y), r, (0, 255, 0), 2)

# Hiển thị ảnh kết quả
cv2.imshow("Detected Lines and Circles", image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

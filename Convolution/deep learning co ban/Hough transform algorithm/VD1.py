import cv2
import numpy as np

# Read the input image
image = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv2.Canny(image, 50, 150, apertureSize=3)

# Perform Hough Line Transform
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
print(lines)
# Draw detected lines on the image
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # The following lines were missing a multiplication symbol in the line equation
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        # Draw the line on the original image
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the output image with detected lines
cv2.imshow('Detected Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np
import cv2
import sobel_operator as sb

image = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
image_gray = sb.magnitude(image)
# image_gray = cv2.Canny(image, 50, 150, apertureSize=3)
cv2.imshow("gray", image_gray)
lines = cv2.HoughLines(image_gray, 1, np.pi / 180, 140)
print(lines)
for line in lines:
    rho, theta = line[0]
    x0 = rho * np.cos(theta)
    y0 = rho * np.sin(theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x1 = int(x0 - 1000 * b)
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 + 1000 * b)
    y2 = int(y0 - 1000 * a)
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow("Hough line image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

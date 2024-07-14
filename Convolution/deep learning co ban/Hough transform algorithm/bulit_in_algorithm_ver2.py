import cv2
import numpy as np
import sobel_operator


def HoughLines(image, threshold):
    r = int(np.ceil(np.sqrt(np.square(image.shape[0]) + np.square(image.shape[1]))))
    result = np.zeros((2 * r, 180))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > 0:
                for k in range(180):
                    print("1")
                    ro1 = int(i * np.cos(np.deg2rad(k)) + j * np.sin(np.deg2rad(k)))
                    print(ro1)
                    result[ro1 + r][k] += 1
    return np.argwhere(result >= threshold)


image = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
image_gray = sobel_operator.magnitude(image)

cv2.imshow("original image", image_gray)
lines = HoughLines(image_gray, 100)
rho = int(np.ceil(np.sqrt(np.square(image_gray.shape[0]) + np.square(image_gray.shape[1]))))

for line in lines:
    ro, theta = line
    ro = ro - rho
    x0, y0 = ro * np.cos(np.deg2rad(theta)), ro * np.sin(np.deg2rad(theta))
    x1 = int(x0 - 1000 * np.sin(np.deg2rad(theta)))
    y1 = int(y0 + 1000 * np.cos(np.deg2rad(theta)))
    x2 = int(x0 + 1000 * np.sin(np.deg2rad(theta)))
    y2 = int(y0 - 1000 * np.cos(np.deg2rad(theta)))
    cv2.line(image_gray, (x1, y1), (x2, y2), (255, 0, 0), 2)
cv2.imshow("filtered image", image_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

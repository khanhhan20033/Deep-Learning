import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
print(image)
sobel_horizontal = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

sobel_vertical = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])


def convolve(image, kernel):
    result = np.zeros((image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1))
    ix = 0
    iy = 0
    for i in range(0 + int((kernel.shape[0] - 1) / 2), image.shape[0] - int((kernel.shape[0] - 1) / 2)):

        for j in range(0 + int((kernel.shape[0] - 1) / 2), image.shape[1] - int((kernel.shape[0] - 1) / 2)):
            result[ix, iy] = np.sum(image[i - int((kernel.shape[0] - 1) / 2): i + int((kernel.shape[0] - 1) / 2) + 1,
                                    j - int((kernel.shape[0] - 1) / 2): j + int(
                                        (kernel.shape[0] - 1) / 2) + 1] * kernel)
            print(result[ix, iy])
            iy += 1
        ix += 1
        iy = 0
    return result


gradient_x = convolve(image, sobel_horizontal).astype(np.uint8)
print(gradient_x[gradient_x > 0])
gradient_y = convolve(image, sobel_vertical).astype(np.uint8)
magnitude = np.sqrt(gradient_y ** 2 + gradient_x ** 2).astype(np.uint8)
print(magnitude)
# print(image.shape)
''''
plt.title('Original_image')
plt.imshow(image, cmap='gray')
plt.show()
plt.title('horizontal gradient')
plt.imshow(gradient_x, cmap='gray')
plt.show()
plt.title('vertical gradient')
plt.imshow(gradient_y, cmap='gray')
plt.show()
plt.title('After applying sobel')
plt.imshow(magnitude, cmap='gray')
plt.show()
'''
cv2.imshow("121", magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()

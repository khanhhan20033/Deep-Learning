import matplotlib.pyplot as plt
import numpy as np
import cv2

sobel_horizontal = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

sobel_vertical = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])


def convolve(image, kernel, threshold):
    result = np.zeros((image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1))
    ix = 0
    iy = 0
    for i in range(0 + int((kernel.shape[0] - 1) / 2), image.shape[0] - int((kernel.shape[0] - 1) / 2)):

        for j in range(0 + int((kernel.shape[0] - 1) / 2), image.shape[1] - int((kernel.shape[0] - 1) / 2)):
            result[ix][iy] = np.sum(image[i - int((kernel.shape[0] - 1) / 2): i + int((kernel.shape[0] - 1) / 2) + 1,
                                    j - int((kernel.shape[0] - 1) / 2): j + int(
                                        (kernel.shape[0] - 1) / 2) + 1] * kernel)
            #if result[ix][iy] < threshold:
                #result[ix][iy] = 0
            iy += 1
        ix += 1
        iy = 0
    return result


def gradient_x(image):
    return convolve(image, sobel_horizontal, 0)


def gradient_y(image):
    return convolve(image, sobel_vertical, 0)


def magnitude(image):
    return np.sqrt(gradient_y(image) ** 2 + gradient_x(image) ** 2).astype(np.uint8)

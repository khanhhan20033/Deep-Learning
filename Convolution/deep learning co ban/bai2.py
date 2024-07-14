import cv2
import numpy as np

# Đọc ảnh
image = cv2.imread('../dangky.png')

# Define các kernel
identity_kernel = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])

edge_detection_kernel = np.array([[-1, -1, -1],
                                  [-1,  8, -1],
                                  [-1, -1, -1]])

sharpen_kernel = np.array([[ 0, -1,  0],
                            [-1,  5, -1],
                            [ 0, -1,  0]])

blur_kernel = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]]) / 9

# Áp dụng phép tích chập với các kernel tương ứng
identity_image = cv2.filter2D(image, -1, identity_kernel)
edge_detection_image = cv2.filter2D(image, -1, edge_detection_kernel)
sharpen_image = cv2.filter2D(image, -1, sharpen_kernel)
blur_image = cv2.filter2D(image, -1, blur_kernel)

# Hiển thị ảnh kết quả
cv2.imshow("Original Image", image)
cv2.imshow("Identity Filter", identity_image)
cv2.imshow("Edge Detection Filter", edge_detection_image)
cv2.imshow("Sharpen Filter", sharpen_image)
cv2.imshow("Blur Filter", blur_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

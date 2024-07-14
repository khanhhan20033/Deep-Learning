import numpy as np
import cv2


def hough_transform(img):
    # Edge detection
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # Define the accumulator matrix
    height, width = edges.shape
    diag_length = int(np.ceil(np.sqrt(height ** 2 + width ** 2)))
    accumulator = np.zeros((2 * diag_length, 250), dtype=np.uint8)

    # Voting
    for y in range(height):
        for x in range(width):
            if edges[y, x] > 0:
                print(f"X:{x}")
                print(f"y:{y}")
                for theta in range(0, 180):
                    rho = int(round(x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta))))
                    print(rho)
                    accumulator[rho + diag_length, theta] += 1

    # Find peaks in the accumulator matrix
    threshold = 100
    loc = np.where(accumulator > threshold)
    for r, t in zip(*loc):
        a = np.cos(np.deg2rad(t))
        b = np.sin(np.deg2rad(t))
        x0 = a * r - diag_length * np.cos(np.deg2rad(t))
        y0 = b * r - diag_length * np.sin(np.deg2rad(t))
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img


# Read an image
image = cv2.imread('1.png')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Hough Transform
result = hough_transform(gray_image)

# Display the result
cv2.imshow('Hough Lines', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

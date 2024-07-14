import numpy as np
import math

def convolution(X, s, p, K):
    """
    Performs a 2D convolution operation.

    Parameters:
    - X: Input 2D array (image).
    - s: Stride of the convolution operation.
    - p: Padding applied to the input array.
    - K: Kernel/filter used for the convolution.

    Returns:
    - Y: Result of the convolution operation.
    """
    # Pad the input image with zeros based on the padding parameter
    X = np.pad(X, pad_width=p)
    
    # Calculate the dimensions of the output feature map
    w, h = math.ceil((X.shape[0] - K.shape[0] + 1) / s), math.ceil((X.shape[1] - K.shape[0] + 1) / s)
    Y = np.zeros((int(w), int(h)))
    
    ii, jj = 0, 0
    # Iterate over the input image with the specified stride
    for i in range(int((K.shape[0] - 1) / 2), int(X.shape[0] - K.shape[0] / 2 + 1 / 2), s):
        for j in range(int((K.shape[0] - 1) / 2), int(X.shape[1] - K.shape[0] / 2 + 1 / 2), s):
            # Apply the kernel/filter and compute the convolution operation
            Y[ii][jj] += np.sum(X[int(i - (K.shape[0] - 1) / 2): int(i + (K.shape[0] + 1) / 2),
                                int(j - (K.shape[0] - 1) / 2): int(j + (K.shape[0] + 1) / 2)] * K)
            jj += 1
        ii += 1
        if jj == int(w):
            jj = 0

    return Y

def backpropagation_convolve_X(X_, grad_y, K, s, p, kernel):
    """
    Computes the gradient of the loss with respect to the input image during backpropagation.

    Parameters:
    - X_: Original 2D array (input image).
    - grad_y: Gradient of the loss with respect to the output feature map.
    - K: Kernel/filter used for the convolution.
    - s: Stride of the convolution operation.
    - p: Padding applied to the input array.
    - kernel: Kernel/filter used for the forward pass.

    Returns:
    - result: Gradient of the loss with respect to the input image.
    """
    # Pad the input image with zeros based on the padding parameter
    X = np.pad(X_, p)
    
    # Initialize an array to hold the result of the convolution operation
    result = np.zeros((X.shape[0] + K - 1, X.shape[1] + K - 1))
    
    idx = 0
    idy = 0
    # Populate the result array with gradients
    for i in range(K - 1, result.shape[0] - K + 1, s):
        for j in range(K - 1, result.shape[1] - K + 1, s):
            result[i, j] = grad_y[idx, idy]
            idy += 1
        idx += 1
        idy = 0
    
    # Create the flipped kernel for the convolution
    fake_kernel = kernel.copy()
    for i in range(fake_kernel.shape[0]):
        fake_kernel[i, :] = fake_kernel[i, ::-1]
    for i in range(fake_kernel.shape[1]):
        fake_kernel[:, i] = fake_kernel[::-1, i]
    
    # Perform the convolution operation to get the gradient with respect to the input
    result = convolution(result, 1, 0, fake_kernel)
    
    return result

def backpropagation_convolve_W(X_, grad_y, K, s, p):
    """
    Computes the gradient of the loss with respect to the kernel during backpropagation.

    Parameters:
    - X_: Original 2D array (input image).
    - grad_y: Gradient of the loss with respect to the output feature map.
    - K: Size of the kernel/filter.
    - s: Stride of the convolution operation.
    - p: Padding applied to the input array.

    Returns:
    - new_kernel: Gradient of the loss with respect to the kernel/filter.
    """
    # Pad the input image with zeros based on the padding parameter
    X = np.pad(X_, p)
    
    # Initialize an array to hold the gradient of the kernel
    new_kernel = np.zeros((X.shape[0] - K + 1, X.shape[1] - K + 1))
    
    idx, idy = 0, 0
    # Populate the new_kernel array with gradients
    for i in range(0, new_kernel.shape[0], s):
        for j in range(0, new_kernel.shape[1], s):
            new_kernel[i, j] = grad_y[idx, idy]
            idy += 0
        idx += 1
        idy = 0
    
    # Compute the gradient of the loss with respect to the kernel
    return convolution(X, s, p, new_kernel)

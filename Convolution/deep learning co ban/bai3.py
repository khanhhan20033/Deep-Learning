import math

import numpy as np


def convolution(X, s, p, K):
    X = np.pad(X, pad_width=p)
    w, h = math.ceil((X.shape[0] - K.shape[0] + 1) / s), math.ceil((X.shape[1] - K.shape[0] + 1) / s)
    Y = np.zeros((int(w), int(h)))
    #print(Y)
    ii, jj = 0, 0
    for i in range(int((K.shape[0] - 1) / 2), int(X.shape[0] - K.shape[0] / 2 + 1 / 2), s):
        # print(f"i:{i}")
        #print(f"ii:{ii}")
        for j in range(int((K.shape[0] - 1) / 2), int(X.shape[1] - K.shape[0] / 2 + 1 / 2), s):
            # print(f"j:{j}")
            #print(f"jj:{jj}")
            # print(X[int(i - (K.shape[0] - 1) / 2): int(i + (K.shape[0] + 1) / 2), int(j - (K.shape[0] - 1) / 2): int(j + (K.shape[0] + 1) / 2)])
            Y[ii][jj] += np.sum(X[int(i - (K.shape[0] - 1) / 2): int(i + (K.shape[0] + 1) / 2),
                                int(j - (K.shape[0] - 1) / 2): int(j + (K.shape[0] + 1) / 2)] * K)
            jj += 1
        ii += 1
        #print(w)
        if jj == int(w):
            jj = 0

    return Y


X = np.array(
    [[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0],
     [0, 1, 1, 0, 0]])
K = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
print(convolution(X, 2, 1, K))

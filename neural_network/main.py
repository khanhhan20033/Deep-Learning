import math

import numpy as np


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def gradient(s):
    return sigmoid(s) * (sigmoid(s) - 1)


def gradient_loss(s):
    return -(s / sigmoid(s) - (1 - s) / (1 - sigmoid(s)))


"""
def neural_network(hidden_layer, W, l, X, b):
    Z = []
    A = []
    for i in range(hidden_layer):
        Z.append([])
        A.append([])
    for i in range(l[0]):

        a = 0
        for j in range(len(X)):
            a += X[j] * W[0][j][i] + b[0][j]
        Z[0].append(a)
        A[0].append(sigmoid(a))
    for i in range(1, hidden_layer):
        for j in range(l[i]):
            a = 0
            for k in range(l[i - 1]):
                a += A[i - 1][k] * W[i][k][j] + b[i][k]
            Z[i].append(a)
            A[i].append(sigmoid(a))
    y = 0
    for i in range(l[hidden_layer - 1]):
        y += A[hidden_layer - 1][i] * W[hidden_layer][i][0] + b[hidden_layer][i]
    A.append(sigmoid(y))
    Z.append(y)
    return A,Z,b
"""


def neural_network(hidden_layer, W, l, X, b):
    Z = [sigmoid(X)]
    A = [X]
    for i in range(hidden_layer + 1):
        Z.append(A[i].dot(W[i]) + b[i])
        A.append(sigmoid(Z[-1]))
    return A, Z


def backpropagation(W, A):
    # gradient_losses = [gradient_loss[A[-1]]]
    gradient_A = []
    gradient_b = []
    gradient_W = []

    for i in range(len(A) - 1, -1, -1):
        if i != 0 and i != len(A) - 1:
            # print(gradient_A[-1])
            #print(gradient(A[i]))
            #print(W[i].shape)
            gradient_A.append((gradient_A[- 1] * gradient(A[i+1])).dot(W[i].T))
        if i == len(A) - 1:
            gradient_A.append(gradient_loss(A[i]))
        if i == 0:
            gradient_A.append(A[i])
        #print(gradient_A)
       # print(i)
        gradient_b.append(gradient_A[-1] * gradient(A[i]))
        gradient_W.append(A[i-1].T.dot(gradient_A[- 1] * gradient(A[i])))



    return gradient_A, gradient_W, gradient_b

#X=np.random.random(20)
X = np.random.random((1, 20))
hidden_layer = int(input("nhap so lop an:"))
l = []
for i in range(hidden_layer):
    l.append(int(input(f"nhap so node cua lop {i + 1}:")))
l.append(1)
# print(l)
W = [np.random.random((X.shape[1], l[0]))]
b = [np.random.random((1, l[0]))]
for i in range(1, hidden_layer + 1):
    w = np.random.random((l[i - 1], l[i]))
    W.append(w)
    c = np.random.random((1, l[i]))
    b.append(c)
# print(b)
A, Z = neural_network(hidden_layer, W, l, X, b)
for i in range(hidden_layer + 2):
    print(f"A[{i}]={A[i]}")
    # print(A[-1])
    print(f"Z[{i}]={Z[i]}")
    # print()
gradient_A, gradient_W, gradient_b = backpropagation(W, A)
print("Cac dao ham theo A:")
for i in range(hidden_layer + 2):
    print(f"lop thu {hidden_layer + 1 - i}: {gradient_A[i]}")
print("Cac dao ham theo W:")
for i in range(hidden_layer + 1):
    print(f"lop thu {hidden_layer - i}: {gradient_A[i]}")

print("Cac dao ham theo b:")
for i in range(hidden_layer + 1):
    print(f"lop thu {hidden_layer - i}: {gradient_b[i]}")

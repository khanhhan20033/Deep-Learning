# import numpy as np
#
#
# def sigmoid(s):
#     return 1 / (1 + np.exp(-s))
#
#
# def sigmoid_grad(s):
#     return s * (1 - s)
#
#
# class Neural_Network:
#
#     def __init__(self, layers):
#         self.layers = layers
#         self.b = []
#         self.W = []
#         for i in range(len(layers) - 1):
#             self.W.append(np.random.randn(layers[i], layers[i + 1]))
#             self.b.append(np.random.randn(layers[i + 1]))
#
#     def model_summary(self):
#         for i in range(len(self.layers)):
#             print(f"layer thu {i + 1} co {self.layers[i]} node")
#
#     def fit_partial(self, X, y, lr=0.1):
#         A = [X]
#         for i in range(self.layers - 1):
#             Z_ = A[i].dot(self.W[i]) + self.b[i]
#             A_ = sigmoid(Z_)
#             A.append(A_)
#         dA = [-(y / A[-1]) + (1 - y) / (1 - A[-1])]
#         dW = []
#         db = []
#         for i in range(range(self.layers)[-2], 1, -1):
#             dA_ = (dA[-1] * sigmoid_grad(A[i])).dot(self.W[i].T)
#             dW_ = A[i].T.dot(dA[-1] * sigmoid_grad(A[i]))
#             db_ = dA[-1] * sigmoid_grad(A[i])
#             dA.append(dA_)
#             dW.append(dW_)
#             db.append(db_)
#         dA.reverse()
#         dW.reverse()
#         db.reverse()
#         for i in range(len(self.layers) - 1):
#             A[i + 1] -= lr * dA[i + 1]
#             self.b[i] -= lr * db[i]
#             self.W[i] -= lr * dW[i]
#
#     def train(self, X, y, lr=0.1, epoch=100, verbose=10, tol=1e-5):
#         for i in range(epoch):
#             self.fit_partial(X, y, lr)
#             if (i + 1) % verbose == 0:
#                 print(f"epoch{i+1}: Loss: {self.calculate_loss(X, y)}")
#             if (np.linalg.norm(np.array(self.W)) / np.array(self.W).size <= tol and
#                     np.linalg.norm(np.array(self.b)) / np.array(self.b).size <= tol):
#                 break
#
#     def pred(self, X):
#         A = [X]
#         for i in range(self.layers - 1):
#             Z_ = A[i].dot(self.W[i]) + self.b[i]
#             A_ = sigmoid(Z_)
#             A.append(A_)
#         return A[-1]
#
#     def calculate_loss(self, X, y):
#         y_ = self.pred(X)
#         return -np.sum(y*np.log(y_)+(1-y)*np.log(1-y_))


import numpy as np


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def sigmoid_derivative(y):
    return sigmoid(y) * (1 - sigmoid(y))


class Neural_network:
    def __init__(self, n_layers):  # [2,2,1]
        self.Z = None
        self.A = None
        self.dW = None
        self.db = None
        self.dA = None
        self.n_layers = n_layers
        self.w = []
        self.b = []
        for i in range(len(n_layers) - 1):
            self.w.append(np.random.randn(n_layers[i], n_layers[i + 1]))
            self.b.append(np.zeros((1, n_layers[i + 1])))
        self.b.insert(0, np.zeros((1, n_layers[0])))

    def fit_partial(self, X, y, lr=0.01):
        self.A = [X]
        # print(self.A)
        self.Z = []
        self.w.insert(0, np.random.randn(X.shape[1], self.n_layers[0]))
        # print(self.w)
        for i in range(len(self.n_layers)):
            #  print(i)
            Z_ = np.dot(self.A[-1], self.w[i])
            self.Z.append(Z_)
            self.A.append(sigmoid(Z_))
        self.dA = [(self.A[-1] - y) / (y * (1 - self.A[-1]))]

        self.db = []
        self.dW = []

        for i in range(len(self.n_layers) - 1, -1, -1):
            self.db.append(np.sum(self.dA[-1] * sigmoid_derivative(self.A[i + 1]), axis=0))
            self.dW.append(self.A[i].T.dot(self.dA[-1] * sigmoid_derivative(self.A[i + 1])))
            self.dA.append((self.dA[-1] * sigmoid_derivative(self.A[i + 1])).dot(self.w[i].T))

        self.db.reverse()
        self.dW.reverse()
        self.dA.reverse()

        for i in range(len(self.n_layers)):
            self.A[i + 1] -= lr * self.dA[i + 1]
            self.b[i] -= lr * self.db[i]
            self.w[i] -= lr * self.dW[i]

    def loss(self, y):
        print(self.A[-1].shape)
        print(y.shape)
        return -y * np.log(self.A[-1]) - (1 - y) * np.log(1 - self.A[-1])

    def fit(self, X, y, epoch=100, verbose=10):
        for i in range(epoch):
            self.fit_partial(X, y)
            self.w.pop(0)
            if epoch % verbose == 0:
                print(f"epoch:{i + 1} - loss:{self.loss(y)[0][0]}")


nn = Neural_network([2, 2, 1])
X = np.arange(30).reshape(1, -1)
y = np.array([[1]])
nn.fit(X, y)

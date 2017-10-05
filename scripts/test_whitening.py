import numpy as np
import scipy.linalg
import matplotlib.pyplot as plot

# The larger the value, the less the transform is applied.
# The worse your data, the smaller should be your lambda.

def whitening_transform(X, lambda_, rotate=True):
    C = (1 / X.shape[0]) * np.dot(X.T, X)
    s, V = scipy.linalg.eigh(C)
    D = np.diag(1. / np.sqrt(s + lambda_))
    W = np.dot(V, D)
    if rotate:
        W = np.dot(W, V.T)
    return W


def whiten(X, mu, W):
    return np.dot(X - mu, W)


features = np.random.normal(0.0, 2.0, size=(100, 2))
W = whitening_transform(features, 1e-3)
w_features = whiten(features, features.mean(), W)

plot.scatter(features[:, 0], features[:, 1], c='blue')
plot.scatter(w_features[:, 0], w_features[:, 1], c='red')
plot.show()

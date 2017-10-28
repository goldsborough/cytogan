import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.datasets.samples_generator import make_blobs


def draw_ellipse(center, covariance, ax):
    # Convert covariance to principal axes
    U, s, _ = np.linalg.svd(covariance)
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    width, height = 2 * np.sqrt(s)
    for nsig in range(1, 4):
        ax.add_patch(
            Ellipse(center, nsig * width, nsig * height, angle, alpha=0.25))


def em(x, k, iterations=10):
    assert np.ndim(x) == 2
    n, features = x.shape
    spans = x.max(axis=0) - x.min(axis=0)

    means = np.random.random([k, features]) * spans
    sigmas = np.random.random([k, features, features]) + np.eye(features)

    for _ in range(iterations):
        # Compute probability for each sample of stemming from each Gaussian.
        probs = np.zeros([n, k])
        for i in range(k):
            p = multivariate_normal.pdf(x, mean=means[i], cov=sigmas[i])
            probs[:, i] = p
        probs /= probs.sum(axis=1, keepdims=True)
        probs /= probs.sum(axis=0, keepdims=True)

        means = np.zeros_like(means)
        for i in range(k):
            p = probs[:, i].reshape(-1, 1)
            means[i] = (p * x).sum(axis=0)

        sigmas = np.zeros_like(sigmas)
        for i in range(k):
            p = probs[:, i].reshape(-1, 1)
            for j in range(n):
                y = (x[j] - means[i]).reshape(features, 1)
                sigmas[i] += p[j] * np.dot(y, y.T)

    return means, sigmas


k = 4
t = 50
x, _ = make_blobs(n_samples=1000, centers=k, cluster_std=1.2)

means, sigmas = em(x, k, t)

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(x[:, 0], x[:, 1], s=5)
for m, s in zip(means, sigmas):
    draw_ellipse(m, s, ax)
ax.autoscale()
plt.show()

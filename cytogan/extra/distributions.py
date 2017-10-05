import numpy as np


def categorical(number_of_classes):
    distribution = [1.0 / number_of_classes] * number_of_classes

    def sample(size):
        samples = np.random.multinomial(1, distribution, size=size)
        if np.ndim(samples) == 2:
            return samples
        return samples.reshape(len(samples), -1)

    return sample


def normal(mean=0.0, stddev=1.0):
    return lambda size: np.random.normal(mean, stddev, size)


def uniform(low=-1.0, high=+1.0):
    return lambda size: np.random.uniform(low, high, size)


def mixture(distribution_count_map):
    def sample(size):
        parts = [d((size, n)) for d, n in distribution_count_map.items()]
        return np.concatenate(parts, axis=1)

    return sample

import numpy as np


def categorical(number_of_classes):
    distribution = [1.0 / number_of_classes] * number_of_classes
    return lambda size: np.random.multinomial(1, distribution, size=size)

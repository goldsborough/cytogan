import keras.backend as K
import numpy as np

E = 1e-10  # numerical stability


def binary_crossentropy(p, q):
    with K.name_scope('binary_crossentropy'):
        pointwise = p * K.log(E + q) + (1 - p) * K.log(E + 1 - q)
        value = -K.mean(pointwise)
    return value


def squared_error(p, q):
    return K.sum(K.square(p - q), axis=1)


def reconstruction_loss(original_images, reconstructed_images):
    if len(original_images.shape) > 2:
        flat_shape = [-1, int(np.prod(original_images.shape[1:]))]
        original_images = K.reshape(original_images, flat_shape)
        reconstructed_images = K.reshape(reconstructed_images, flat_shape)
    return squared_error(original_images, reconstructed_images)

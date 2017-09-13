import keras.backend as K
import numpy as np

E = 1e-10  # numerical stability


def binary_crossentropy(p, q):
    return K.mean(K.binary_crossentropy(p, q))


def squared_errors(p, q):
    '''MSE(p, q) = ||p - q||^2'''
    return K.sum(K.square(p - q), axis=1)


def reconstruction_loss(original_images, reconstructed_images):
    if len(original_images.shape) > 2:
        flat_shape = [-1, int(np.prod(original_images.shape[1:]))]
        original_images = K.reshape(original_images, flat_shape)
        reconstructed_images = K.reshape(reconstructed_images, flat_shape)
    return squared_errors(original_images, reconstructed_images)


def mutual_information(x, x_given_y):
    '''
    I(x;y) = H(x) - H(x|y).
    NOTE: We return the negative mutual information as a suitable minimization
    target.
    '''
    # The cross entropy between x and x is just the entropy H(x).
    h_x = K.categorical_crossentropy(x, x)
    # The cross entropy between x and x|y is H(x|y).
    h_x_given_y = K.categorical_crossentropy(x, x_given_y)
    # The mutual information I(x;y) is now H(x) - H(x|y).
    # Usually we want to maximize mutual information, but to provide a
    # minimizable objective for TF's optimizer, we return E[-(H(x) -
    # H(x|y))] = E[H(x|y) - H(x).]
    return K.mean(h_x_given_y - h_x)

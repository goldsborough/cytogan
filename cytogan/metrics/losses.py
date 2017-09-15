import keras.backend as K
import numpy as np
import tensorflow as tf

E = 1e-10  # numerical stability


def binary_crossentropy(p, q):
    p = K.flatten(p)
    q = K.flatten(q)
    return K.mean(K.binary_crossentropy(p, q), axis=-1)


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
    with K.name_scope('mutual_information'):
        # The cross entropy between x and x is just the entropy H(x).
        h_x = K.categorical_crossentropy(x, x)
        # The cross entropy between x and x|y is H(x|y).
        h_x_given_y = K.categorical_crossentropy(x, x_given_y)
        # The mutual information I(x;y) is now H(x) - H(x|y).
        # Usually we want to maximize mutual information, but to provide a
        # minimizable objective for TF's optimizer, we return E[-(H(x) -
        # H(x|y))] = E[H(x|y) - H(x).]
        return K.mean(h_x_given_y - h_x)


def mixed_mutual_information(x, x_given_y, discrete_continuous_split):
    discrete_x = x[:discrete_continuous_split]
    discrete_x_given_y = x_given_y[:discrete_continuous_split]
    discrete_mi = mutual_information(discrete_x, discrete_x_given_y)

    continuous_x = x[discrete_continuous_split:]
    continuous_mean, continuous_sigma = tf.split(
        x_given_y[discrete_continuous_split:], 2)
    continuous_nll = 0

    return discrete_mi + continuous_nll

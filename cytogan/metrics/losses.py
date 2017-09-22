import keras.backend as K
import numpy as np
import tensorflow as tf
import keras.losses

E = 1e-10  # numerical stability


def l1_distance(x, y):
    x = K.flatten(x)
    y = K.flatten(y)
    return keras.losses.mean_absolute_error(x, y)


def squared_error(p, q):
    '''MSE(p, q) = ||p - q||^2'''
    return K.sum(K.square(p - q), axis=1)


def reconstruction_loss(original_images, reconstructed_images):
    if len(original_images.shape) > 2:
        flat_shape = [-1, int(np.prod(original_images.shape[1:]))]
        original_images = K.reshape(original_images, flat_shape)
        reconstructed_images = K.reshape(reconstructed_images, flat_shape)
    return squared_error(original_images, reconstructed_images)


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


def log_likelihood(p, mean, log_variance):
    '''Negative log likelihood of a Gaussian-distributed variable.'''
    # http://docs.chainer.org/en/stable/reference/generated/chainer.functions.gaussian_nll.html#chainer.functions.gaussian_nll
    epsilon = K.square(p - mean) * K.exp(-log_variance)
    pointwise = 0.5 * (K.log(2 * np.pi) + log_variance + epsilon)
    return K.mean(K.sum(pointwise, axis=1))


def mixed_mutual_information(x, x_given_y, discrete_continuous_split,
                             continuous_lambda):
    discrete_prior = x[:, :discrete_continuous_split]
    discrete_posterior = x_given_y[:, :discrete_continuous_split]
    discrete_mi = mutual_information(discrete_prior, discrete_posterior)

    if discrete_continuous_split == x.shape[1]:
        return discrete_mi

    # Compute likelihood of prior under unit Gaussian.
    continuous_prior = x[:, discrete_continuous_split:]
    prior_likelihood = log_likelihood(continuous_prior, 0.0, 0.0)

    # Compute likelihood of prior under estimated Gaussian parameters.
    mean, log_variance = tf.split(
        x_given_y[:, discrete_continuous_split:], 2, axis=1)
    posterior_likelihood = log_likelihood(continuous_prior, mean, log_variance)
    continuous_likelihood_difference = posterior_likelihood - prior_likelihood

    return discrete_mi + continuous_lambda * continuous_likelihood_difference

import keras.backend as K
import numpy as np
import tensorflow as tf
import keras.losses

E = 1e-10  # numerical stability


def cosine_distance(u, v):
    if len(u.shape) == 1:
        u = tf.expand_dims(u, axis=0)
    if len(v.shape) == 1:
        v = tf.expand_dims(v, axis=1)
    else:
        v = tf.transpose(v)

    u_norm = tf.norm(u, axis=1, keep_dims=True)
    v_norm = tf.norm(v, axis=0, keep_dims=True)
    norm = tf.matmul(u_norm, v_norm)
    dot = tf.matmul(u, v)
    assert norm.shape.as_list() == dot.shape.as_list(), (norm, dot)

    cosine_similarity = dot / (norm + E)
    cosine_distance = tf.reduce_mean(1 - cosine_similarity)

    return tf.squeeze(cosine_distance)


def l1_distance(x, y):
    with K.name_scope('l1_distance'):
        x = K.flatten(x)
        y = K.flatten(y)
        return keras.losses.mean_absolute_error(x, y)


def binary_crossentropy(p, q):
    with K.name_scope('bce'):
        p = K.flatten(p)
        q = K.flatten(q)
        return keras.losses.binary_crossentropy(p, q)


def mean_squared_error(p, q):
    with K.name_scope('mse'):
        p = K.flatten(p)
        q = K.flatten(q)
        return keras.losses.mean_squared_error(p, q)


def squared_error(p, q):
    '''MSE(p, q) = ||p - q||^2'''
    return K.sum(K.square(p - q), axis=1)


def reconstruction_loss(original_images, reconstructed_images):
    if len(original_images.shape) > 2:
        flat_shape = [-1, int(np.prod(original_images.shape[1:]))]
        original_images = K.reshape(original_images, flat_shape)
        reconstructed_images = K.reshape(reconstructed_images, flat_shape)
    return squared_error(original_images, reconstructed_images)


def mutual_information(x,
                       x_given_y,
                       cross_entropy_function=K.categorical_crossentropy):
    '''
    I(x;y) = H(x) - H(x|y).
    NOTE: We return the negative mutual information as a suitable minimization
    target.
    '''
    assert x.shape.as_list() == x_given_y.shape.as_list(), (x, x_given_y)
    with K.name_scope('mutual_info'):
        # The cross entropy between x and x is just the entropy H(x).
        h_x = cross_entropy_function(x, x)
        # The cross entropy between x and x|y is H(x|y).
        h_x_given_y = cross_entropy_function(x, x_given_y)
        # The mutual information I(x;y) is now H(x) - H(x|y).
        # Usually we want to maximize mutual information, but to provide a
        # minimizable objective for TF's optimizer, we return E[-(H(x) -
        # H(x|y))] = E[H(x|y) - H(x).]
        return K.mean(h_x_given_y - h_x)


def binary_mutual_information(x, x_given_y):
    with K.name_scope('binary_mutual_info'):
        return mutual_information(x, x_given_y, K.binary_crossentropy)


def log_likelihood(p, mean, log_variance):
    '''Negative log likelihood of a Gaussian-distributed variable.'''
    # http://docs.chainer.org/en/stable/reference/generated/chainer.functions.gaussian_nll.html#chainer.functions.gaussian_nll
    with K.name_scope('log_likelihood'):
        epsilon = K.square(p - mean) * K.exp(-log_variance)
        pointwise = 0.5 * (K.log(2 * np.pi) + log_variance + epsilon)
        return K.mean(K.sum(pointwise, axis=1))


def log_likelihood_difference(x, x_given_y):
    assert x_given_y.shape[1] == x.shape[1] * 2, (x, x_given_y)
    # Compute likelihood of prior under unit Gaussian.
    prior_likelihood = log_likelihood(x, 0.0, 0.0)

    # Compute likelihood of prior under estimated Gaussian parameters.
    mean, log_variance = tf.split(x_given_y, 2, axis=1)
    assert mean.shape[1] == log_variance.shape[1] == x.shape[1]
    posterior_likelihood = log_likelihood(x, mean, log_variance)

    return posterior_likelihood - prior_likelihood


def mixed_mutual_information(x, x_given_y, discrete_continuous_split,
                             continuous_lambda, continuous_loss):
    if discrete_continuous_split > 0:
        discrete_prior = x[:, :discrete_continuous_split]
        discrete_posterior = x_given_y[:, :discrete_continuous_split]
        discrete_mi = mutual_information(discrete_prior, discrete_posterior)
    else:
        discrete_mi = 0

    if discrete_continuous_split == x.shape[1]:
        return discrete_mi

    continuous_prior = x[:, discrete_continuous_split:]
    continuous_posterior = x_given_y[:, discrete_continuous_split:]

    if continuous_loss == 'bce':
        continuous_loss_value = binary_mutual_information(
            continuous_prior, continuous_posterior)
    else:
        continuous_loss_value = log_likelihood_difference(
            continuous_prior, continuous_posterior)

    return discrete_mi + continuous_lambda * continuous_loss_value

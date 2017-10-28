import keras.backend as K
import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                          Dense, Flatten, Input, LeakyReLU, Reshape, Lambda,
                          UpSampling2D)
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.examples.tutorials.mnist import input_data

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(session)

# Supress warnings about wrong compilation of TensorFlow.
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(42)

noise_size = 100

latent_discrete = 10
latent_continuous = 2
latent_size = latent_discrete + latent_continuous

from keras.layers import Layer


class BatchNorm(Layer):
    def __init__(self, axis=-1, momentum=0.9, variance_epsilon=1e-3, **kwargs):
        self.axis = axis
        self.momentum = momentum
        self.variance_epsilon = variance_epsilon

        self.scale = None
        self.offset = None
        self.population_mean = None
        self.population_variance = None

        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = (input_shape[self.axis], )
        with K.name_scope('batch_norm'):
            self.scale = self.add_weight(
                shape=shape, name='scale', initializer='ones')
            self.offset = self.add_weight(
                shape=shape, name='offset', initializer='zeros')

            self.population_mean = self.add_weight(
                shape=shape,
                name='population_mean',
                initializer='zeros',
                trainable=False)
            self.population_variance = self.add_weight(
                shape=shape,
                name='population_variance',
                initializer='ones',
                trainable=False)

        super(BatchNorm, self).build(input_shape)

    def call(self, inputs, training=None):
        with K.name_scope('batch_norm'):
            if training in {0, False}:
                return self._test_graph(inputs)
            else:
                return self._training_graph(inputs)
            # return tf.cond(K.learning_phase(),
            #                lambda: self._training_graph(inputs),
            #                lambda: self._test_graph(inputs))

    def compute_output_shape(self, input_shape):
        return input_shape

    def _training_graph(self, inputs):
        reduction_axes = list(range(len(inputs.shape)))
        reduction_axes.pop(self.axis)

        mean, variance = tf.nn.moments(
            inputs, axes=reduction_axes, name='moments')
        variance = tf.maximum(variance, tf.constant(0.0))

        updated_mean = self._mix(self.population_mean, mean)
        update_population_mean = tf.assign(self.population_mean, updated_mean)

        updated_variance = self._mix(self.population_variance, variance)
        update_population_variance = tf.assign(self.population_variance,
                                               updated_variance)

        with tf.control_dependencies(
            [update_population_mean, update_population_variance]):
            return tf.nn.batch_normalization(
                inputs,
                mean,
                variance,
                self.offset,
                self.scale,
                variance_epsilon=self.variance_epsilon)

    def _test_graph(self, inputs):
        return tf.nn.batch_normalization(
            inputs,
            self.population_mean,
            self.population_variance,
            self.offset,
            self.scale,
            variance_epsilon=self.variance_epsilon)

    def _mix(self, old, new):
        return self.momentum * old + (1 - self.momentum) * new


## G

z = Input(shape=[noise_size])
c = Input(shape=[latent_size])
G = Concatenate()([z, c])

G = Dense(7 * 7 * 256)(G)
G = BatchNorm()(G)
# G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)
G = Reshape((7, 7, 256))(G)

G = UpSampling2D()(G)
G = Conv2D(128, (5, 5), padding='same')(G)
G = BatchNorm()(G)
# G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)

G = UpSampling2D()(G)
G = Conv2D(64, (5, 5), padding='same')(G)
G = BatchNorm()(G)
# G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)

G = Conv2D(32, (5, 5), padding='same')(G)
G = BatchNorm()(G)
# G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)

G = Conv2D(1, (5, 5), padding='same')(G)
G = Activation('tanh')(G)

## D

x = Input(shape=(28, 28, 1))
D = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(x)
D = LeakyReLU(alpha=0.2)(D)

D = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(D)
D = LeakyReLU(alpha=0.2)(D)

D = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(D)
D = LeakyReLU(alpha=0.2)(D)

D = Conv2D(256, (5, 5), padding='same')(D)
D = LeakyReLU(alpha=0.2)(D)
D = Flatten()(D)


def latent_activations(Q):
    Q_discrete = Activation('softmax')(Q[:, :latent_discrete])
    Q_continuous = Activation('tanh')(Q[:, latent_discrete:])
    return Concatenate(axis=1)([Q_discrete, Q_continuous])


Q = Dense(latent_discrete + 2 * latent_continuous)(D)
Q = Lambda(latent_activations)(Q)

P = Dense(1, activation='sigmoid')(D)


def mutual_information(prior_c, c_given_x):
    h_c = K.categorical_crossentropy(prior_c, prior_c)
    h_c_given_x = K.categorical_crossentropy(prior_c, c_given_x)
    return K.mean(h_c_given_x - h_c)


def log_likelihood(p, mean, log_variance):
    epsilon = K.square(p - mean) * K.exp(-log_variance)
    pointwise = 0.5 * (K.log(2 * np.pi) + log_variance + epsilon)
    return K.mean(K.sum(pointwise, axis=1))


def joint_mutual_information(prior_c, c_given_x):
    discrete = mutual_information(prior_c[:, :latent_discrete],
                                  c_given_x[:, :latent_discrete])

    prior_continuous = prior_c[:, -latent_continuous:]
    prior_likelihood = log_likelihood(prior_continuous, 0.0, 0.0)

    mean, log_variance = tf.split(c_given_x[:, latent_discrete:], 2, axis=1)
    posterior_likelihood = log_likelihood(prior_continuous, mean, log_variance)
    continuous = posterior_likelihood - prior_likelihood

    return discrete + 0.8 * continuous


generator = Model([z, c], G, name='G')

discriminator = Model(x, P, name='D')
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=5e-4, beta_1=0.5, decay=2e-7))

# x = G(z, c)
q = Model(x, Q, name='Q')
q.compile(
    loss=joint_mutual_information,
    optimizer=Adam(lr=2e-4, beta_1=0.5, decay=2e-7))

discriminator.trainable = False
q.trainable = False
infogan = Model([z, c], [discriminator(G), q(G)], name='InfoGAN')
infogan.compile(
    loss=['binary_crossentropy', joint_mutual_information],
    optimizer=Adam(lr=2e-4, beta_1=0.5, decay=1e-7))

generator.summary()
discriminator.summary()

data = input_data.read_data_sets('MNIST_data').train.images
data = data.reshape(-1, 28, 28, 1) * 2 - 1

number_of_epochs = 30
batch_size = 128


def sample_noise(size):
    return np.random.randn(size, noise_size)


def sample_prior(size):
    discrete = np.random.multinomial(1, [0.1] * 10, size=size)
    continuous_1 = np.random.uniform(-1, +1, size).reshape(-1, 1)
    continuous_2 = np.random.uniform(-1, +1, size).reshape(-1, 1)
    return np.concatenate([discrete, continuous_1, continuous_2], axis=1)


def smooth_labels(size):
    return np.random.uniform(low=0.8, high=1.0, size=size)


try:
    for epoch in range(number_of_epochs):
        print('Epoch: {0}/{1}'.format(epoch + 1, number_of_epochs))
        for batch_start in range(0, len(data) - batch_size + 1, batch_size):
            noise = sample_noise(batch_size)
            latent_code = sample_prior(batch_size)
            generated_images = generator.predict([noise, latent_code])

            real_images = data[batch_start:batch_start + batch_size]
            assert len(generated_images) == len(real_images)
            all_images = np.concatenate(
                [generated_images, real_images], axis=0)
            all_images += np.random.normal(0, 0.1, all_images.shape)

            labels = np.zeros(len(all_images))
            labels[batch_size:] = smooth_labels(batch_size)
            d_loss = discriminator.train_on_batch(all_images, labels)

            q_loss = q.train_on_batch(generated_images, latent_code)

            labels = np.ones(batch_size)
            noise = sample_noise(batch_size)
            latent_code = sample_prior(batch_size)
            g_loss, _, _ = infogan.train_on_batch([noise, latent_code],
                                                  [labels, latent_code])

            batch_index = batch_start // batch_size + 1
            message = '\rBatch: {0} | D: {1:.10f} | G: {2:.10f} | Q: {3:.10f}'
            print(message.format(batch_index, d_loss, g_loss, q_loss), end='')
        print()
        np.random.shuffle(data)
except KeyboardInterrupt:
    print()

print('Training complete!')


def plot_digits(filename, with_c1=False, with_c2=False):
    images_per_digit = 20
    total_images = 10 * images_per_digit
    if with_c1 or with_c2:
        noise = sample_noise(1).repeat(total_images, axis=0)
    else:
        noise = np.tile(sample_noise(images_per_digit), (10, 1))

    discrete = np.eye(latent_discrete).repeat(images_per_digit, axis=0)
    if with_c1:
        continuous_1 = np.linspace(-3, +3, images_per_digit).reshape(-1, 1)
        continuous_1 = np.tile(continuous_1, (10, 1))
    else:
        continuous_1 = np.zeros([total_images, 1])
    if with_c2:
        continuous_2 = np.linspace(-3, +3, images_per_digit).reshape(-1, 1)
        continuous_2 = np.tile(continuous_2, (10, 1))
    else:
        continuous_2 = np.zeros([total_images, 1])
    latent_code = np.concatenate(
        [discrete, continuous_1, continuous_2], axis=1)
    images = generator.predict_on_batch([noise, latent_code])
    images = (images + 1) / 2
    plot.switch_backend('Agg')
    plot.figure(figsize=(10, 4))
    for i in range(total_images):
        axis = plot.subplot(10, images_per_digit, i + 1)
        plot.imshow(images[i].reshape(28, 28), cmap='gray')
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
    print('Saving {0}'.format(filename))
    plot.savefig(filename)


plot_digits('fig_d.png')
plot_digits('fig_c1.png', with_c1=True)
plot_digits('fig_c2.png', with_c2=True)

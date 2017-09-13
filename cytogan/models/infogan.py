import collections

import keras.backend as K
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                          Dense, Flatten, Input, LeakyReLU, Reshape,
                          UpSampling2D)
from keras.models import Model
from keras.optimizers import Adam

from cytogan.models import model
from cytogan.metrics import losses

import tensorflow as tf
import numpy as np

Hyper = collections.namedtuple('Hyper', [
    'image_shape',
    'generator_filters',
    'discriminator_filters',
    'generator_strides',
    'discriminator_strides',
    'latent_size',
    'noise_size',
    'initial_shape',
    'latent_priors',
])


def _get_labels(fake_images, real_images):
    fake_labels = np.zeros(len(fake_images))
    # github.com/soumith/ganhacks#6-use-soft-and-noisy-labels
    real_labels = np.random.uniform(low=0.8, high=1.0, size=len(real_images))
    return np.concatenate((fake_labels, real_labels)).reshape(-1, 1)


class InfoGAN(model.Model):
    def __init__(self, hyper, learning, session):
        assert len(hyper.image_shape) == 3
        # Copy all fields from hyper to self.
        for index, field in enumerate(hyper._fields):
            setattr(self, field, hyper[index])

        self.noise = None  # z

        self.latent_prior = None  # c
        self.latent_posterior = None  # c|x

        self.images = None  # x
        self.labels = None  # 0/1

        self.probability = None  # D(x)

        self.generator = None  # G(z, c)
        self.discriminator = None  # D(x)
        self.encoder = None  # Q(c|x)
        self.infogan = None  # D(G(z, c)) + Q(G(z, c))

        super(InfoGAN, self).__init__(learning, session)

    def _define_graph(self):
        with K.name_scope('G'):
            tensors = self._define_generator()
        self.noise, self.latent_prior, self.fake_images = tensors

        self.images, logits = self._define_discriminator()

        self.latent_posterior = Dense(
            self.latent_size, activation='softmax', name='Q_final')(logits)
        self.probability = Dense(
            1, activation='sigmoid', name='D_final')(logits)

        self.generator = Model(
            [self.noise, self.latent_prior], self.fake_images, name='G')

        self.loss = {}
        self.labels = Input(shape=[1], name='labels')

        self.discriminator = Model(self.images, self.probability, name='D')
        with K.name_scope('D_loss'):
            self.loss['D'] = losses.binary_crossentropy(
                self.labels, self.probability)

        self.encoder = Model(self.images, self.latent_posterior, name='Q')
        with K.name_scope('Q_loss'):
            self.loss['Q'] = losses.mutual_information(self.latent_prior,
                                                       self.latent_posterior)

        self.infogan = Model(
            [
                self.noise,
                self.latent_prior,
            ], [
                self.discriminator(self.fake_images),
                self.encoder(self.fake_images),
            ],
            name='InfoGAN')
        with K.name_scope('G_loss'):
            self.infogan_bce = losses.binary_crossentropy(
                self.labels, self.infogan.outputs[0])
            self.infogan_mi = losses.mutual_information(
                self.latent_prior, self.infogan.outputs[1])
            self.loss['G'] = self.infogan_bce + self.infogan_mi

        assert all(len(l.shape) == 0 for l in self.loss.values()), self.loss

    @property
    def learning_rate(self):
        learning_rates = {}
        for key, lr in self._learning_rate.items():
            if isinstance(lr, tf.Tensor):
                lr = lr.eval(session=self.session)
            learning_rates[key] = lr
        return learning_rates

    def encode(self, images):
        return self.session.run(
            self.latent_posterior, feed_dict={self.images: images})

    def generate(self, latent_samples):
        if isinstance(latent_samples, int):
            number_of_samples = latent_samples
            latent_samples = self._sample_priors(number_of_samples)
        else:
            number_of_samples = len(latent_samples)
        noise = self._sample_noise(number_of_samples)
        images = self.session.run(
            self.fake_images,
            feed_dict={
                self.noise: noise,
                self.latent_prior: latent_samples,
                K.learning_phase(): 0,
            })

        # Go from [-1, +1] scale back to [0, 1]
        return (images + 1) / 2

    def train_on_batch(self, real_images, with_summary=False):
        batch_size = len(real_images)
        real_images = (real_images * 2) - 1
        noise = self._sample_noise(batch_size)
        latent_code = self._sample_priors(batch_size)

        fake_images = self.session.run(
            self.fake_images,
            feed_dict={
                self.noise: noise,
                self.latent_prior: latent_code,
                K.learning_phase(): 0,
            })

        d_loss = self._train_discriminator(fake_images, real_images)
        q_loss = self._train_encoder(fake_images, latent_code)
        g_tensors = self._train_generator(batch_size, with_summary)

        losses = dict(D=d_loss, G=g_tensors[0], Q=q_loss)
        return (losses, g_tensors[1]) if with_summary else losses

    def _train_discriminator(self, fake_images, real_images):
        assert len(fake_images) == len(real_images)
        labels = _get_labels(fake_images, real_images)
        assert labels.shape == (2 * len(real_images), 1)

        images = np.concatenate([fake_images, real_images], axis=0)
        # github.com/soumith/ganhacks#13-add-noise-to-inputs-decay-over-time
        images += np.random.normal(0, 0.1, images.shape)

        # L_D = -D(x) -D(G(z, c))
        _, discriminator_loss = self.session.run(
            [self.optimizer['D'], self.loss['D']],
            feed_dict={
                self.images: images,
                self.labels: labels,
                K.learning_phase(): 1,
            })

        return discriminator_loss

    def _train_encoder(self, fake_images, latent_prior):
        # I(c; G(z, c))
        _, encoder_loss = self.session.run(
            [self.optimizer['Q'], self.loss['Q']],
            feed_dict={
                self.images: fake_images,
                self.latent_prior: latent_prior,
                K.learning_phase(): 1,
            })

        return encoder_loss

    def _train_generator(self, batch_size, with_summary):
        noise = self._sample_noise(batch_size)
        latent_code = self._sample_priors(batch_size)
        fetches = [self.optimizer['G'], self.loss['G']]
        if with_summary:
            fetches.append(self.summary)

        results = self.session.run(
            fetches,
            feed_dict={
                self.noise: noise,
                self.latent_prior: latent_code,
                self.labels: np.ones([batch_size, 1]),
                K.learning_phase(): 1,
            })

        return results[1:]

    def _add_summaries(self):
        super(InfoGAN, self)._add_summaries()
        tf.summary.histogram('noise', self.noise)
        tf.summary.histogram('latent_prior', self.latent_prior)
        tf.summary.histogram('probability', self.infogan.outputs[0])
        tf.summary.histogram('latent_posterior', self.infogan.outputs[1])
        tf.summary.scalar('G_bce', self.infogan_bce)
        tf.summary.scalar('G_mi', self.infogan_mi)
        tf.summary.scalar('G_loss', self.loss['G'])
        tf.summary.image('generated_images', self.fake_images, max_outputs=4)

    def _define_generator(self):
        z = Input(shape=[self.noise_size], name='z')
        c = Input(shape=[self.latent_size], name='c')
        G = Concatenate()([z, c])

        first_filter = self.generator_filters[0]
        G = Dense(np.prod(self.initial_shape) * first_filter)(G)
        G = BatchNormalization(momentum=0.9)(G)
        G = LeakyReLU(alpha=0.2)(G)
        G = Reshape(self.initial_shape + self.generator_filters[:1])(G)

        for filters, stride in zip(self.generator_filters[1:],
                                   self.generator_strides[1:]):
            if stride > 1:
                G = UpSampling2D(stride)(G)
            G = Conv2D(filters, (5, 5), padding='same')(G)
            G = BatchNormalization(momentum=0.9)(G)
            G = LeakyReLU(alpha=0.2)(G)

        G = Conv2D(1, (5, 5), padding='same')(G)
        G = Activation('tanh')(G)
        assert G.shape[1:] == self.image_shape, G.shape

        return z, c, G

    def _define_discriminator(self):
        x = Input(shape=self.image_shape, name='images')
        D = x
        for filters, stride in zip(self.discriminator_filters,
                                   self.discriminator_strides):
            D = Conv2D(
                filters, (5, 5), strides=(stride, stride), padding='same')(D)
            D = LeakyReLU(alpha=0.2)(D)
        D = Flatten()(D)

        with tf.control_dependencies([tf.assert_positive(D, [D])]):
            return x, D

    def _add_optimizer(self, learning):
        self.optimizer = {}
        self._learning_rate = {}
        initial_learning_rate = learning.rate
        if isinstance(initial_learning_rate, float):
            initial_learning_rate = [initial_learning_rate] * 3

        self._learning_rate['D'] = self._get_learning_rate_tensor(
            initial_learning_rate[0], learning.decay, learning.steps_per_decay)
        self.optimizer['D'] = tf.train.AdamOptimizer(
            self._learning_rate['D'], beta1=0.5).minimize(self.loss['D'])

        self._learning_rate['Q'] = self._get_learning_rate_tensor(
            initial_learning_rate[1], learning.decay, learning.steps_per_decay)
        self.optimizer['Q'] = tf.train.AdamOptimizer(
            self._learning_rate['Q'], beta1=0.5).minimize(self.loss['Q'])

        generator_weights = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
        assert all(w.name.startswith('G/') for w in generator_weights)
        self._learning_rate['G'] = self._get_learning_rate_tensor(
            initial_learning_rate[2], learning.decay, learning.steps_per_decay)
        self.optimizer['G'] = tf.train.AdamOptimizer(
            self._learning_rate['G'], beta1=0.5).minimize(
                self.loss['G'],
                var_list=generator_weights,
                global_step=self.global_step)

    def _sample_noise(self, size):
        return np.random.randn(size, self.noise_size)

    def _sample_priors(self, number_of_samples):
        if isinstance(self.latent_priors, collections.Iterable):
            return np.concatenate(
                [p(number_of_samples) for p in self.latent_priors], axis=1)
        return self.latent_priors(number_of_samples)

    def __repr__(self):
        lines = [self.__class__.__name__]
        try:
            # >= Keras 2.0.6
            self.generator.summary(print_fn=lines.append)
            self.discriminator.summary(print_fn=lines.append)
        except TypeError:
            lines = [layer.name for layer in self.generator.layers]
            lines = [layer.name for layer in self.discriminator.layers]
        return '\n'.join(map(str, lines))

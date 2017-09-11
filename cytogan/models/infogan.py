import collections

import keras.backend as K
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                          Dense, Flatten, Input, LeakyReLU, Reshape,
                          UpSampling2D)
from keras.models import Model

from cytogan.models import model
from cytogan.metrics import losses

import tensorflow as tf
import numpy as np

Hyper = collections.namedtuple('Hyper', [
    'image_shape',
    'filter_sizes',
    'latent_size',
    'noise_size',
    'rescales',
    'initial_shape',
    'latent_priors',
])


def _get_labels(fake_images, real_images):
    fake_labels = np.zeros(len(fake_images))
    # github.com/soumith/ganhacks#6-use-soft-and-noisy-labels
    real_labels = np.random.uniform(low=0.8, high=1.0, size=len(real_images))
    return np.concatenate((fake_labels, real_labels))


class InfoGAN(model.Model):
    def __init__(self, hyper, learning, session):
        assert len(hyper.image_shape) == 3
        # Copy all fields from hyper to self.
        for index, field in enumerate(hyper._fields):
            setattr(self, field, hyper[index])

        self.noise = None  # z

        self.latent_prior = None  # c
        self.latent_predicted = None  # c|x

        self.images = None  # x
        self.labels = None  # 0/1

        self.probability = None  # D(x)

        self.generator = None  # G(z, c)
        self.generator_loss = None

        self.discriminator = None  # D(x)
        self.discriminator_loss = None

        self.encoder = None  # Q(c|x)
        self.encoder_loss = None

        super(InfoGAN, self).__init__([learning], session)

    def _define_graph(self):
        tensors = self._define_generator()
        self.noise, self.latent_prior, self.fake_images = tensors
        self.images, logits = self._define_discriminator()

        self.latent_predicted = Dense(
            self.latent_size, activation='softmax')(logits)
        self.probability = Dense(1, activation='sigmoid')(logits)

        self.generator = Model(
            [self.noise, self.latent_prior], self.fake_images, name='G')

        self.labels = Input(batch_shape=[None])
        self.discriminator = Model(self.images, self.probability, name='D')

        with tf.control_dependencies([tf.assert_positive(self.probability)]):
            self.discriminator_loss = losses.binary_crossentropy(
                self.labels, self.probability)

        self.encoder = Model(self.images, self.latent_predicted, name='Q')
        self.encoder_loss = losses.mutual_information(self.latent_prior,
                                                      self.latent_predicted)

        # Disable weights when reusing layers here because updating the whole
        # model should just train the generator.
        self.discriminator.trainable = False
        self.encoder.trainable = False
        infogan = Model(
            inputs=[self.noise, self.latent_prior],
            outputs=[
                self.discriminator(self.fake_images),
                self.encoder(self.fake_images)
            ],
            name='InfoGAN')
        mi = losses.mutual_information(self.latent_prior, infogan.outputs[1])
        self.infogan_loss = -K.log(infogan.outputs[0]) + mi

        return dict(
            G=self.infogan_loss,
            Q=self.encoder_loss,
            D=self.discriminator_loss)

    def encode(self, images):
        return self.session.run(
            self.latent_predicted, feed_dict={self.images: images})

    def generate(self, latent_samples):
        if isinstance(latent_samples, int):
            number_of_samples = latent_samples
            latent_samples = self._sample_priors(number_of_samples)
        else:
            number_of_samples = len(latent_samples)
        noise = self._sample_noise(number_of_samples)
        return self.session.run(
            self.generator.output,
            feed_dict={
                self.latent_prior: latent_samples,
                self.noise: noise,
                K.learning_phase(): 0,
            })

    def train_on_batch(self, real_images, with_summary=False):
        discriminator_loss = self._train_discriminator(real_images)

        noise = self._sample_noise(len(real_images))
        latent = self._sample_priors(len(real_images))
        fetches = [self.optimizer['G'], self.infogan_loss]
        if with_summary:
            fetches.append(self.summary)

        # L_G = -D(G(z, c))
        results = self.session.run(
            fetches,
            feed_dict={
                self.latent_prior: latent,
                self.noise: noise,
                K.learning_phase(): 1,
            })

        losses = dict(D=discriminator_loss, G=results[1])
        return (losses, results[2]) if with_summary else losses

    def _add_summaries(self):
        tf.summary.histogram('noise', self.noise)
        tf.summary.histogram('latent_prior', self.latent_prior)
        tf.summary.histogram('latent_predicted', self.latent_predicted)
        tf.summary.histogram('probability', self.probability)
        tf.summary.image('generated_images', self.fake_images, max_outputs=4)
        super(InfoGAN, self)._add_summaries()

    def _define_generator(self):
        with tf.name_scope('G'):
            z = Input(shape=[self.noise_size])
            c = Input(shape=[self.latent_size])
            G = Concatenate()([z, c])

            G = Dense(np.prod(self.initial_shape) * self.filter_sizes[0])(G)
            G = BatchNormalization(momentum=0.9)(G)
            G = LeakyReLU(alpha=0.2)(G)
            G = Reshape(self.initial_shape + self.filter_sizes[:1])(G)

            for filters, rescale in zip(self.filter_sizes[1:],
                                        self.rescales[1:]):
                if rescale > 1:
                    G = UpSampling2D(rescale)(G)
                G = Conv2D(filters, (5, 5), padding='same')(G)
                G = BatchNormalization(momentum=0.9)(G)
                G = LeakyReLU(alpha=0.2)(G)

            G = Conv2D(1, (5, 5), padding='same')(G)
            G = Activation('tanh')(G)
            assert G.shape[1:] == self.image_shape, G.shape

        return z, c, G

    def _define_discriminator(self):
        with tf.name_scope('D'):
            x = Input(shape=(28, 28, 1))
            D = x
            for filters, scale in zip(self.filter_sizes[::-1],
                                      self.rescales[::-1]):
                D = Conv2D(filters, (5, 5), strides=scale, padding='same')(D)
                D = LeakyReLU(alpha=0.2)(D)
            D = Flatten()(D)

        return x, D

    def _train_discriminator(self, real_images):
        latent = self._sample_priors(len(real_images))
        assert latent.shape == real_images.shape[:1] + (self.latent_size, )
        fake_images = self.generate(latent)
        images = np.concatenate([fake_images, real_images], axis=0)
        # github.com/soumith/ganhacks#13-add-noise-to-inputs-decay-over-time
        images += np.random.normal(0, 0.1, images.shape)
        labels = _get_labels(fake_images, real_images)

        # L_D = -D(x) -D(G(z, c))
        _, discriminator_loss = self.session.run(
            [self.optimizer['D'], self.discriminator_loss],
            feed_dict={
                self.images: images,
                self.labels: labels,
                K.learning_phase(): 1,
            })

        # L_D = -D(x) -D(G(z, c)) + I(c; G(z, c))
        _, mutual_information_loss = self.session.run(
            [self.optimizer['Q'], self.encoder_loss],
            feed_dict={
                self.images: fake_images,
                self.latent_prior: latent,
                K.learning_phase(): 1,
            })

        return discriminator_loss + mutual_information_loss

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

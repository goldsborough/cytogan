import collections

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Concatenate, Conv2D, Dense, Dropout,
                          Flatten, Input, LeakyReLU, Reshape, UpSampling2D)
from keras.models import Model

from cytogan.extra.layers import AddNoise, BatchNorm
from cytogan.metrics import losses
from cytogan.models import gan, util

tf.logging.set_verbosity(tf.logging.INFO)

Hyper = collections.namedtuple('Hyper', [
    'image_shape',
    'generator_filters',
    'generator_strides',
    'encoder_filters',
    'encoder_strides',
    'discriminator_filters',
    'discriminator_strides',
    'latent_size',
    'initial_shape',
])


class BiGAN(gan.GAN):
    def __init__(self, hyper, learning, session):
        self.noise_size = hyper.latent_size
        super(BiGAN, self).__init__(hyper, learning, session)

    def encode(self, images, rescale=True):
        if rescale:
            images = (images * 2.0) - 1

        return self.session.run(self.latent, {
            K.learning_phase(): 0,
            self.images_to_encode: images
        })

    def reconstruct(self, images, rescale=True):
        latent = self.encode(images, rescale)
        return self.generate(latent, rescale)

    def generate(self, latent_samples, rescale=True):
        if isinstance(latent_samples, int):
            latent_samples = self._sample_noise(latent_samples)
        images = self.session.run(self.fake_images, {
            K.learning_phase(): 0,
            self.noise: latent_samples
        })

        # Go from [-1, +1] scale back to [0, 1]
        return (images + 1) / 2.0 if rescale else images

    def _define_graph(self):
        with K.name_scope('G'):
            self.noise = Input(shape=[self.noise_size], name='noise')
            self.fake_images = self._define_generator(self.noise)

        with K.name_scope('E'):
            self.images_to_encode = Input(
                shape=self.image_shape, name='real_images')
            self.latent = self._define_encoder(self.images_to_encode)

        with K.name_scope('D'):
            self.images = Input(shape=self.image_shape, name='images')
            self.input_code = Input(
                shape=[self.latent_size], name='representations')
            self.probability = self._define_discriminator(
                self.images, self.input_code)

        self.labels = Input(batch_shape=[None], name='labels')

        self.generator = Model(self.noise, self.fake_images, name='G')
        self.encoder = Model(self.images_to_encode, self.latent, name='E')
        self.discriminator = Model(
            [self.images, self.input_code], self.probability, name='D')
        self.generator_gan = Model(
            self.noise,
            self.discriminator([self.fake_images, self.noise]),
            name=self.name + '-G')
        self.encoder_gan = Model(
            self.images_to_encode,
            self.discriminator([self.images_to_encode, self.latent]),
            name=self.name + '-E')

        self.loss = dict(
            D=self._define_discriminator_loss(self.labels, self.probability),
            G=self._define_generator_loss(self.generator_gan.outputs[0]),
            E=self._define_encoder_loss(self.encoder_gan.outputs[0]))

    def _define_generator(self, noise):
        first_filter = self.generator_filters[0]
        G = Dense(np.prod(self.initial_shape) * first_filter)(noise)
        G = BatchNorm()(G)
        G = LeakyReLU(alpha=0.2)(G)
        G = Reshape(self.initial_shape + self.generator_filters[:1])(G)

        for filters, stride in zip(self.generator_filters[1:],
                                   self.generator_strides[1:]):
            if stride > 1:
                G = UpSampling2D(stride)(G)
            G = Conv2D(filters, (5, 5), padding='same')(G)
            G = BatchNorm()(G)
            G = LeakyReLU(alpha=0.2)(G)

        G = Conv2D(self.number_of_channels, (5, 5), padding='same')(G)
        G = Activation('tanh')(G)
        assert G.shape[1:] == self.image_shape, G.shape

        return G

    def _define_encoder(self, images):
        E = AddNoise()(images)

        for filters, stride in zip(self.encoder_filters, self.encoder_strides):
            E = Conv2D(
                filters, (5, 5), strides=(stride, stride), padding='same')(E)
            E = BatchNorm()(E)
            E = LeakyReLU(alpha=0.2)(E)

        E = Flatten()(E)
        E = Dense(self.latent_size, name='latent')(E)

        return E

    def _define_discriminator(self, images, code):
        D = AddNoise()(images)
        for filters, stride in zip(self.discriminator_filters[0],
                                   self.discriminator_strides):
            D = Conv2D(
                filters, (5, 5), strides=(stride, stride), padding='same')(D)
            D = LeakyReLU(alpha=0.2)(D)
            D = Dropout(0.5)(D)
        D = Flatten()(D)

        D = Concatenate()([D, code])

        for units in self.discriminator_filters[1]:
            D = Dense(units)(D)
            D = LeakyReLU(alpha=0.2)(D)
            D = Dropout(0.5)(D)

        D = Dense(1, activation='sigmoid')(D)

        return D

    def _define_generator_loss(self, probability):
        with K.name_scope('G_loss'):
            ones = K.ones_like(probability)
            return losses.binary_crossentropy(ones, probability)

    def _define_encoder_loss(self, probability):
        with K.name_scope('E_loss'):
            ones = K.zeros_like(probability)
            return losses.binary_crossentropy(ones, probability)

    def _define_discriminator_loss(self, labels, probability):
        labels = gan.smooth_labels(labels)
        with K.name_scope('D_loss'):
            return losses.binary_crossentropy(labels, probability)

    def train_on_batch(self, batch, with_summary=False):
        real_images = (np.array(batch) * 2.0) - 1

        noise = self._sample_noise(len(real_images))
        fake_images = self.generate(noise, rescale=False)
        real_code = self.encode(real_images, rescale=False)

        d_tensors = self._train_discriminator(fake_images, real_images, noise,
                                              real_code, with_summary)

        noise = self._sample_noise(len(real_images))
        g_tensors = self._train_generator(noise, with_summary)

        e_tensors = self._train_encoder(real_images, with_summary)

        losses = dict(D=d_tensors[0], G=g_tensors[0], E=e_tensors[0])
        tensors = dict(G=g_tensors, D=d_tensors, E=e_tensors)
        return self._maybe_with_summary(losses, tensors, with_summary)

    def _train_discriminator(self, fake_images, real_images, fake_code,
                             real_code, with_summary):
        labels = util.binary_labels(len(fake_images), len(real_images))
        images = np.concatenate([fake_images, real_images], axis=0)
        code = np.concatenate([fake_code, real_code], axis=0)

        fetches = [self.optimizer['D'], self.loss['D']]
        if with_summary:
            fetches.append(self.summaries['D'])

        outputs = self.session.run(fetches, {
            self.images: images,
            self.input_code: code,
            self.labels: labels,
            K.learning_phase(): 1,
        })

        return outputs[1:]

    def _train_generator(self, noise, with_summary):
        fetches = [self.optimizer['G'], self.loss['G']]
        if with_summary:
            fetches.append(self.summaries['G'])

        outputs = self.session.run(fetches, {
            self.noise: noise,
            K.learning_phase(): 1,
        })

        return outputs[1:]

    def _train_encoder(self, images, with_summary):
        fetches = [self.optimizer['E'], self.loss['E']]
        if with_summary:
            fetches.append(self.summaries['E'])

        outputs = self.session.run(fetches, {
            self.images_to_encode: images,
            K.learning_phase(): 1,
        })

        return outputs[1:]

    def _add_optimizer(self, learning):
        assert isinstance(learning.rate, list), 'lr must be list of 3 floats'
        assert len(learning.rate) == 3, 'lr must be list of 3 floats for BiGAN'
        super(BiGAN, self)._add_optimizer(learning)
        initial_learning_rate = learning.rate

        with K.name_scope('opt/E'):
            self._learning_rate['E'] = self._get_learning_rate_tensor(
                initial_learning_rate[2], learning.decay,
                learning.steps_per_decay)
            self.optimizer['E'] = tf.train.AdamOptimizer(
                self._learning_rate['E'], beta1=0.5).minimize(
                    self.loss['E'], var_list=self.encoder.trainable_weights)

    def _sample_noise(self, size):
        return np.random.randn(size, self.noise_size)

    def _get_summary_nodes(self):
        return {scope: util.merge_summaries(scope) for scope in 'DGE'}

    def _add_summaries(self):
        with K.name_scope('summary/G'):
            tf.summary.histogram('noise', self.noise)
            tf.summary.scalar('loss', self.loss['G'])
            tf.summary.image(
                'generated_images', self.fake_images, max_outputs=4)

        with K.name_scope('summary/D'):
            tf.summary.scalar('loss', self.loss['D'])

        with K.name_scope('summary/E'):
            tf.summary.histogram('latent', self.latent)
            tf.summary.scalar('loss', self.loss['E'])

    def __repr__(self):
        lines = [self.name]
        try:
            # >= Keras 2.0.6
            self.generator.summary(print_fn=lines.append)
            self.encoder.summary(print_fn=lines.append)
            self.discriminator.summary(print_fn=lines.append)
        except TypeError:
            lines = [layer.name for layer in self.generator.layers]
            lines = [layer.name for layer in self.encoder.layers]
            lines = [layer.name for layer in self.discriminator.layers]
        return '\n'.join(map(str, lines))

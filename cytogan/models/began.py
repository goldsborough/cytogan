import collections

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Conv2D, Dense, Flatten, Input, Reshape)
from keras.models import Model

from cytogan.extra.layers import AddNoise, RandomNormal, UpSamplingNN
from cytogan.models import gan
from cytogan.metrics import losses

Hyper = collections.namedtuple('Hyper', [
    'image_shape',
    'generator_filters',
    'encoder_filters',
    'decoder_filters',
    'generator_strides',
    'encoder_strides',
    'decoder_strides',
    'latent_size',
    'noise_size',
    'initial_shape',
    'diversity_factor',
    'proportional_gain',
])


class BEGAN(gan.GAN):
    def __init__(self, hyper, learning, session):
        self.k = None
        self.reconstructions = None
        super(BEGAN, self).__init__(hyper, learning, session)

    def _define_graph(self):
        with K.name_scope('G'):
            self.batch_size = Input(batch_shape=[1], name='batch_size')
            self.noise = RandomNormal(self.noise_size)(self.batch_size)
            self.fake_images = self._define_generator(self.noise)

        self.images = Input(shape=self.image_shape, name='images')

        with K.name_scope('E'):
            self.latent = self._define_encoder(self.images)
        with K.name_scope('D'):
            self.reconstructions = self._define_decoder(self.latent)

        self.generator = Model(self.batch_size, self.fake_images, name='G')
        self.discriminator = Model(self.images, self.reconstructions, name='D')
        self.encoder = Model(self.images, self.latent, name='E')
        self.gan = Model(
            self.batch_size,
            self.discriminator(self.fake_images),
            name=self.name)

        self.loss = dict(
            D=self._define_discriminator_loss(self.reconstructions),
            G=self._define_generator_loss(self.gan.outputs[0]))

    def _train_discriminator(self, fake_images, real_images, with_summary):
        images = np.concatenate([fake_images, real_images], axis=0)
        fetches = [self.optimizer['D'], self.loss['D']]
        if with_summary:
            fetches.append(self.discriminator_summary)
        fetches.append(self.update_k)

        results = self.session.run(
            fetches,
            feed_dict={
                self.batch_size: [len(fake_images)],
                self.images: images,
            })

        return results[1:]

    def _train_generator(self, batch_size, with_summary):
        fetches = [self.optimizer['G'], self.loss['G']]
        if with_summary:
            fetches.append(self.generator_summary)

        results = self.session.run(
            fetches, feed_dict={self.batch_size: [batch_size]})

        return results[1:]

    def _define_generator(self, logits):
        first_filter = self.generator_filters[0]
        initial_flat_shape = np.prod(self.initial_shape) * first_filter
        G = Dense(initial_flat_shape, activation='elu')(logits)
        G = Reshape(self.initial_shape + self.generator_filters[:1])(G)

        for filters, stride in zip(self.generator_filters[1:],
                                   self.generator_strides[1:]):
            if stride > 1:
                G = UpSamplingNN(stride)(G)
            G = Conv2D(filters, (3, 3), padding='same', activation='elu')(G)
            G = Conv2D(filters, (3, 3), padding='same', activation='elu')(G)

        G = Conv2D(self.number_of_channels, (3, 3), padding='same')(G)
        G = Activation('tanh')(G)
        assert G.shape[1:] == self.image_shape, G.shape

        return G

    def _define_encoder(self, images):
        E = AddNoise()(images)
        for filters, stride in zip(self.encoder_filters, self.encoder_strides):
            E = Conv2D(
                filters,
                kernel_size=(3, 3),
                strides=(stride, stride),
                padding='same',
                activation='elu')(E)
            E = Conv2D(128, (3, 3), padding='same', activation='elu')(E)
            E = Conv2D(128, (3, 3), padding='same', activation='elu')(E)

        E = Flatten()(E)
        H = Dense(self.latent_size)(E)

        return H

    def _define_decoder(self, latent_code):
        first_filter = self.decoder_filters[0]
        initial_flat_shape = np.prod(self.initial_shape) * first_filter
        D = Dense(initial_flat_shape)(latent_code)
        D = Reshape(self.initial_shape + self.decoder_filters[:1])(D)
        for filters, stride in zip(self.decoder_filters[1:],
                                   self.decoder_strides[1:]):
            if stride > 1:
                D = UpSamplingNN(stride)(D)
            D = Conv2D(filters, (3, 3), padding='same', activation='elu')(D)
            D = Conv2D(filters, (3, 3), padding='same', activation='elu')(D)

        D = Conv2D(self.number_of_channels, (3, 3), padding='same')(D)
        assert D.shape[1:] == self.image_shape, D.shape

        return D

    def _define_generator_loss(self, logits):
        with K.name_scope('G_loss'):
            return losses.l1_distance(self.fake_images, logits)

    def _define_discriminator_loss(self, reconstructions):
        with K.name_scope('D_loss'):
            with K.name_scope('slicing'):
                batch_size = tf.cast(tf.squeeze(self.batch_size), tf.int32)
                fake_images = self.images[:batch_size]
                real_images = self.images[batch_size:]
                fake_reconstructions = reconstructions[:batch_size]
                real_reconstructions = reconstructions[batch_size:]

            with K.name_scope('real_loss'):
                real_loss = losses.l1_distance(real_images,
                                               real_reconstructions)
            with K.name_scope('fake_loss'):
                fake_loss = losses.l1_distance(fake_images,
                                               fake_reconstructions)

            self.k = tf.Variable(1e-8, trainable=False, name='k')

            with K.name_scope('loss_value'):
                loss = real_loss - (self.k * fake_loss)

            with K.name_scope('equilibrium'):
                equilibrium = (self.diversity_factor * real_loss) - fake_loss

            with K.name_scope('convergence_measure'):
                self.convergence_measure = real_loss + tf.abs(equilibrium)

            with K.name_scope('k_update'):
                self.k_pre_clip = self.k + self.proportional_gain * equilibrium
                new_k = tf.clip_by_value(self.k_pre_clip, 0, 1)
                self.update_k = tf.assign(self.k, new_k)

            return loss

    def _add_summaries(self):
        with K.name_scope('G'):
            tf.summary.histogram('noise', self.noise)
            tf.summary.scalar('G_loss', self.loss['G'])
            tf.summary.image(
                'generated_images', self.fake_images, max_outputs=8)

        with K.name_scope('D'):
            tf.summary.histogram('latent', self.latent)
            tf.summary.scalar('D_loss', self.loss['D'])
            tf.summary.scalar('k_pre_clip', self.k_pre_clip)
            tf.summary.scalar('k', self.k)
            tf.summary.scalar('convergence', self.convergence_measure)
            batch_size = tf.cast(tf.squeeze(self.batch_size), tf.int32)
            fake_reconstructions = self.reconstructions[:batch_size]
            real_reconstructions = self.reconstructions[batch_size:]
            tf.summary.image(
                'fake_reconstructions', fake_reconstructions, max_outputs=2)
            tf.summary.image(
                'real_reconstructions', real_reconstructions, max_outputs=2)

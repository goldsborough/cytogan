import collections

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Concatenate, Conv2D, Dense, Flatten,
                          Input, LeakyReLU, Reshape, UpSampling2D)
from keras.models import Model

from cytogan.extra.layers import (AddNoise, BatchNorm, RandomNormal,
                                  RandomUniform)
from cytogan.metrics import losses
from cytogan.models import gan

Hyper = collections.namedtuple('Hyper', [
    'image_shape',
    'generator_filters',
    'discriminator_filters',
    'generator_strides',
    'discriminator_strides',
    'latent_size',
    'noise_size',
    'initial_shape',
    'conditional_shape',
    'conditional_embedding',
    'noise_kind',
])


class DCGAN(gan.GAN):
    def __init__(self, hyper, learning, session):
        self.batch_size = None
        self.labels = None  # 0/1
        self.d_final = None  # D(x)

        super(DCGAN, self).__init__(hyper, learning, session)

    def _train_discriminator(self, fake_images, real_images, conditional,
                             with_summary):
        labels = np.concatenate(
            [np.zeros(len(fake_images)),
             np.ones(len(real_images))], axis=0)
        images = np.concatenate([fake_images, real_images], axis=0)
        fetches = [self.optimizer['D'], self.loss['D']]
        if with_summary and self.summaries['D'] is not None:
            fetches.append(self.summaries['D'])

        feed_dict = {
            self.batch_size: [len(fake_images)],
            self.images: images,
            self.labels: labels,
            K.learning_phase(): 1,
        }
        if self.is_conditional:
            # Not sure why we need to feed the generator conditional, but TF
            # complains otherwise.
            feed_dict[self.conditional['G']] = np.zeros_like(conditional)
            # Duplicate the conditional (for the real and for the fake images).
            conditional = np.concatenate([conditional, conditional], axis=0)
            feed_dict[self.conditional['D']] = conditional

        return self.session.run(fetches, feed_dict)[1:]

    def _train_generator(self, batch_size, conditional, with_summary):
        fetches = [self.optimizer['G'], self.loss['G']]
        if with_summary and self.summaries['G'] is not None:
            fetches.append(self.summaries['G'])

        feed_dict = {self.batch_size: [batch_size], K.learning_phase(): 1}
        if self.is_conditional:
            feed_dict[self.conditional['G']] = conditional

        return self.session.run(fetches, feed_dict)[1:]

    def _define_graph(self):
        if self.is_conditional:
            self._define_conditional_units(self.conditional_shape)

        with K.name_scope('G'):
            self.batch_size = Input(batch_shape=[1], name='batch_size')
            if self.noise_kind == 'normal':
                self.noise = RandomNormal(self.noise_size)(self.batch_size)
            else:
                self.noise = RandomUniform(self.noise_size)(self.batch_size)
            conditional = self._get_conditional_embedding('G')
            self.fake_images = self._define_generator(self.noise, conditional)

        with K.name_scope('D'):
            self.images = Input(shape=self.image_shape, name='images')
            logits = self._define_discriminator(self.images)
            self.latent = Dense(self.latent_size, name='latent')(logits)
            if self.is_conditional:
                conditional = self._get_conditional_embedding('D')
                final_input = Concatenate(axis=1)([self.latent, conditional])
            else:
                final_input = self.latent
            self.d_final = self._define_final_discriminator_layer(final_input)

        self.labels = Input(batch_shape=[None], name='labels')

        parameters = self._get_model_parameters(self.is_conditional)
        generator_inputs, discriminator_inputs, generator_outputs = parameters

        self.generator = Model(generator_inputs, self.fake_images, name='G')
        self.discriminator = Model(
            discriminator_inputs, self.d_final, name='D')
        self.encoder = Model(discriminator_inputs, self.latent, name='E')
        self.gan = Model(
            generator_inputs,
            self.discriminator(generator_outputs),
            name=self.name)

        self.loss = dict(
            D=self._define_discriminator_loss(self.labels, self.d_final),
            G=self._define_generator_loss(self.gan.outputs[0]))

    def _define_generator(self, noise, conditional=None):
        if conditional is None:
            logits = noise
        else:
            logits = Concatenate(axis=1)([noise, conditional])
        first_filter = self.generator_filters[0]
        G = Dense(np.prod(self.initial_shape) * first_filter)(logits)
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

    def _define_discriminator(self, images):
        D = AddNoise()(images)
        for filters, stride in zip(self.discriminator_filters,
                                   self.discriminator_strides):
            D = Conv2D(
                filters, (5, 5), strides=(stride, stride), padding='same')(D)
            D = LeakyReLU(alpha=0.2)(D)
        D = Flatten()(D)

        return D

    def _define_conditional_units(self, conditional_shape):
        self.conditional = gan.get_conditional_inputs(
            ('G', 'D'), conditional_shape)
        if self.conditional_embedding is not None:
            self.conditional_embedding_layer = Dense(
                self.conditional_embedding,
                activation='relu',
                name='embedding')

    def _define_generator_loss(self, probability):
        with K.name_scope('G_loss'):
            ones = K.ones_like(probability)
            return losses.binary_crossentropy(ones, probability)

    def _define_discriminator_loss(self, labels, probability):
        labels = gan.smooth_labels(labels)
        with K.name_scope('D_loss'):
            return losses.binary_crossentropy(labels, probability)

    def _define_final_discriminator_layer(self, logits):
        return Dense(1, activation='sigmoid', name='Probability')(logits)

    def _add_summaries(self):
        super(DCGAN, self)._add_summaries()
        if self.is_conditional:
            with K.name_scope('summary/G'):
                tf.summary.histogram('conditional', self.conditional['G'])

        with K.name_scope('summary/D'):
            batch_size = tf.cast(tf.squeeze(self.batch_size), tf.int32)
            fake_output = self.d_final[:batch_size]
            real_output = self.d_final[batch_size:]
            tf.summary.histogram('fake_output', fake_output)
            tf.summary.histogram('real_output', real_output)
            if self.is_conditional:
                tf.summary.histogram('conditional', self.conditional['D'])

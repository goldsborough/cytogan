import collections

import keras.backend as K
import keras.losses
import numpy as np
import tensorflow as tf
from keras.layers import Activation, Concatenate, Dense, Input, Lambda
from keras.models import Model

from cytogan.extra.layers import RandomNormal
from cytogan.metrics import losses
from cytogan.models import dcgan

Hyper = collections.namedtuple('Hyper', [
    'image_shape',
    'generator_filters',
    'discriminator_filters',
    'generator_strides',
    'discriminator_strides',
    'latent_size',
    'noise_size',
    'initial_shape',
    'latent_distribution',
    'discrete_variables',
    'continuous_variables',
    'continuous_lambda',
])


class InfoGAN(dcgan.DCGAN):
    def __init__(self, hyper, learning, session):
        self.latent_prior = None  # c
        self.latent_posterior = None  # c|x

        super(InfoGAN, self).__init__(hyper, learning, session)

    def _define_graph(self):
        with K.name_scope('G'):
            self.batch_size = Input(batch_shape=[1], name='batch_size')
            self.noise = RandomNormal(self.noise_size)(self.batch_size)
            self.latent_prior = Input(shape=[self.latent_size])
            full_latent = Concatenate()([self.noise, self.latent_prior])
            self.fake_images = self._define_generator(full_latent)

        with K.name_scope('D'):
            self.images = Input(shape=self.image_shape, name='images')
            logits = self._define_discriminator(self.images)
            self.latent_posterior = Lambda(
                self._latent_layer, name='Q/final')(logits)
            self.probability = Dense(
                1, activation='sigmoid', name='D/final')(logits)

        generator_inputs = [self.batch_size, self.latent_prior]
        self.generator = Model(generator_inputs, self.fake_images, name='G')

        self.loss = {}
        self.labels = Input(batch_shape=[None], name='labels')

        self.discriminator = Model(self.images, self.probability, name='D')
        self.encoder = Model(self.images, self.latent_posterior, name='Q')
        self.gan = Model(
            generator_inputs, [
                self.discriminator(self.fake_images),
                self.encoder(self.fake_images),
            ],
            name=self.name)

        self.loss = dict(
            D=self._define_discriminator_loss(self.probability),
            Q=self._define_encoder_loss(self.encoder.output),
            G=self._define_generator_loss(*self.gan.outputs))

    def generate(self, latent_samples):
        images = self.generator.predict_on_batch(
            [len(latent_samples), latent_samples])

        # Go from [-1, +1] scale back to [0, 1]
        return (images + 1) / 2

    def train_on_batch(self, real_images, with_summary=False):
        real_images = (real_images * 2.0) - 1
        batch_size = len(real_images)

        latent_code = self.latent_distribution(batch_size)
        fake_images = self.generator.predict([batch_size, latent_code])
        assert len(fake_images) == len(real_images)

        all_images = np.concatenate([fake_images, real_images], axis=0)
        all_images += np.random.normal(0, 0.1, all_images.shape)

        d_tensors = self._train_discriminator(fake_images, real_images,
                                              with_summary)
        q_loss = self._train_encoder(fake_images, latent_code)
        g_tensors = self._train_generator(batch_size, with_summary)

        losses = dict(D=d_tensors[0], G=g_tensors[0], Q=q_loss)

        if with_summary:
            summary = self._get_combined_summary(g_tensors[1], d_tensors[1])
            return losses, summary
        else:
            return losses

    def _train_generator(self, batch_size, with_summary):
        latent_code = self.latent_distribution(batch_size)
        fetches = [self.optimizer['G'], self.loss['G']]
        if with_summary:
            fetches.append(self.summary)

        results = self.session.run(
            fetches,
            feed_dict={
                self.batch_size: [batch_size],
                self.latent_prior: latent_code,
                K.learning_phase(): 1,
            })

        return results[1:]

    def _train_encoder(self, fake_images, latent_prior):
        # I(c; G(z, c))
        _, encoder_loss = self.session.run(
            [self.optimizer['Q'], self.loss['Q']],
            feed_dict={
                self.batch_size: [len(fake_images)],
                self.images: fake_images,
                self.latent_prior: latent_prior,
                K.learning_phase(): 1,
            })

        return encoder_loss

    def _add_optimizer(self, learning):
        super(InfoGAN, self)._add_optimizer(learning)
        initial_learning_rate = learning.rate
        if isinstance(initial_learning_rate, float):
            initial_learning_rate = [initial_learning_rate] * 3
        else:
            assert len(initial_learning_rate) == 3

        with K.name_scope('opt/Q'):
            self._learning_rate['Q'] = self._get_learning_rate_tensor(
                initial_learning_rate[2], learning.decay,
                learning.steps_per_decay)
            self.optimizer['Q'] = tf.train.AdamOptimizer(
                self._learning_rate['Q'], beta1=0.5).minimize(
                    self.loss['Q'], var_list=self.encoder.trainable_weights)

    def _latent_layer(self, logits):
        logits = Dense(
            self.discrete_variables + 2 * self.continuous_variables,
            name='Q/final_dense')(logits)
        discrete = Activation('softmax')(logits[:, :self.discrete_variables])
        continuous = Activation('linear')(logits[:, self.discrete_variables:])
        return Concatenate(axis=1)([discrete, continuous])

    def _define_generator_loss(self, probability, latent_posterior):
        with K.name_scope('G/loss'):
            self.infogan_bce = keras.losses.binary_crossentropy(
                K.ones_like(probability), probability)
            self.infogan_mi = losses.mixed_mutual_information(
                self.latent_prior, latent_posterior, self.discrete_variables,
                self.continuous_lambda)

            return self.infogan_bce + self.infogan_mi

    def _define_discriminator_loss(self, probability):
        with K.name_scope('D/loss'):
            return keras.losses.binary_crossentropy(self.labels, probability)

    def _define_encoder_loss(self, latent_posterior):
        with K.name_scope('Q/loss'):
            return losses.mixed_mutual_information(
                self.latent_prior, latent_posterior, self.discrete_variables,
                self.continuous_lambda)

    def _add_summaries(self):
        super(InfoGAN, self)._add_summaries()
        with K.name_scope('summary/G'):
            tf.summary.histogram('latent_prior', self.latent_prior)
        with K.name_scope('summary/D'):
            tf.summary.scalar('bce', self.infogan_bce)
            tf.summary.scalar('mi', self.infogan_mi)
        with K.name_scope('summary/Q'):
            tf.summary.histogram('latent_posterior', self.gan.outputs[1])

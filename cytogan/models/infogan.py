import collections

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Activation, Concatenate, Dense, Input, Lambda
from keras.models import Model

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
            self.noise = Input(shape=[self.noise_size])
            self.latent_prior = Input(shape=[self.latent_size])
            full_latent = Concatenate()([self.noise, self.latent_prior])
            self.fake_images = self._define_generator(full_latent)

        self.images, logits = self._define_discriminator()

        self.latent_posterior = Lambda(
            self._latent_layer, name='Q_final')(logits)
        self.probability = Dense(
            1, activation='sigmoid', name='D_final')(logits)

        self.generator = Model(
            [self.noise, self.latent_prior], self.fake_images, name='G')

        self.loss = {}
        self.labels = Input(batch_shape=[None], name='labels')

        self.discriminator = Model(self.images, self.probability, name='D')
        with K.name_scope('D_loss'):
            self.loss['D'] = losses.binary_crossentropy(
                self.labels, self.discriminator.output)

        self.encoder = Model(self.images, self.latent_posterior, name='Q')
        with K.name_scope('Q_loss'):
            self.loss['Q'] = losses.mixed_mutual_information(
                self.latent_prior, self.encoder.output,
                self.discrete_variables,
                self.continuous_lambda)

        self.gan = Model(
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
                K.ones_like(self.gan.outputs[0]), self.gan.outputs[0])
            self.infogan_mi = losses.mixed_mutual_information(
                self.latent_prior, self.gan.outputs[1],
                self.discrete_variables,
                self.continuous_lambda)
            self.loss['G'] = self.infogan_bce + self.infogan_mi

    def generate(self, latent_samples):
        noise_samples = self._sample_noise(len(latent_samples))
        images = self.generator.predict_on_batch(
            [noise_samples, latent_samples])

        # Go from [-1, +1] scale back to [0, 1]
        return (images + 1) / 2

    def train_on_batch(self, real_images, with_summary=False):
        real_images = (real_images * 2) - 1
        batch_size = len(real_images)
        noise = self._sample_noise(batch_size)
        latent_code = self.latent_distribution(batch_size)
        fake_images = self.generator.predict([noise, latent_code])
        assert len(fake_images) == len(real_images)
        all_images = np.concatenate([fake_images, real_images], axis=0)
        all_images += np.random.normal(0, 0.1, all_images.shape)

        d_loss = self._train_discriminator(fake_images, real_images)
        q_loss = self._train_encoder(fake_images, latent_code)
        g_tensors = self._train_generator(batch_size, with_summary)

        losses = dict(D=d_loss, G=g_tensors[0], Q=q_loss)
        return (losses, g_tensors[1]) if with_summary else losses

    def _add_summaries(self):
        super(InfoGAN, self)._add_summaries()
        tf.summary.histogram('latent_prior', self.latent_prior)
        tf.summary.histogram('latent_posterior', self.gan.outputs[1])
        tf.summary.scalar('G_bce', self.infogan_bce)
        tf.summary.scalar('G_mi', self.infogan_mi)

    def _train_generator(self, batch_size, with_summary):
        noise = self._sample_noise(batch_size)
        latent_code = self.latent_distribution(batch_size)
        fetches = [self.optimizer['G'], self.loss['G']]
        if with_summary:
            fetches.append(self.summary)

        results = self.session.run(
            fetches,
            feed_dict={
                self.noise: noise,
                self.latent_prior: latent_code,
                K.learning_phase(): 0,
            })

        return results[1:]

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

    def _add_optimizer(self, learning):
        super(InfoGAN, self)._add_optimizer(learning)
        initial_learning_rate = learning.rate
        if isinstance(initial_learning_rate, float):
            initial_learning_rate = [initial_learning_rate] * 3
        else:
            assert len(initial_learning_rate) == 3

        with K.name_scope('Q_opt'):
            self._learning_rate['Q'] = self._get_learning_rate_tensor(
                initial_learning_rate[2], learning.decay,
                learning.steps_per_decay)
            self.optimizer['Q'] = tf.train.AdamOptimizer(
                self._learning_rate['Q'], beta1=0.5).minimize(
                    self.loss['Q'], var_list=self.encoder.trainable_weights)

    def _latent_layer(self, logits):
        logits = Dense(
            self.discrete_variables + 2 * self.continuous_variables,
            name='Q_final_dense')(logits)
        discrete = Activation('softmax')(logits[:, :self.discrete_variables])
        continuous = Activation('tanh')(logits[:, self.discrete_variables:])
        return Concatenate(axis=1)([discrete, continuous])

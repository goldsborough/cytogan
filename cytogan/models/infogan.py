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
    return np.concatenate((fake_labels, real_labels))


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
        self.generator_loss = None

        self.discriminator = None  # D(x)
        self.discriminator_loss = None

        self.encoder = None  # Q(c|x)
        self.encoder_loss = None

        self.infogan = None  # D(G(z, c)) + Q(G(z, c))
        self.infogan_loss = None

        super(InfoGAN, self).__init__(learning, session)

    def _define_graph(self):
        tensors = self._define_generator()
        self.noise, self.latent_prior, self.fake_images = tensors
        self.images, logits = self._define_discriminator()

        self.latent_posterior = Dense(
            self.latent_size, activation='softmax')(logits)
        self.probability = Dense(1, activation='sigmoid')(logits)

        self.generator = Model(
            [self.noise, self.latent_prior], self.fake_images, name='G')

        self.labels = Input(batch_shape=[None])

        def mutual_information(prior_c, c_given_x):
            h_c = K.categorical_crossentropy(prior_c, prior_c)
            h_c_given_x = K.categorical_crossentropy(prior_c, c_given_x)
            value = K.mean(h_c_given_x - h_c)
            print(value.shape)
            return value

        self.generator = Model(
            [self.noise, self.latent_prior], self.fake_images, name='G')

        self.discriminator = Model(self.images, self.probability, name='D')
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=Adam(lr=5e-4, beta_1=0.5, decay=2e-7))

        # x = G(z, c)
        self.q = Model(self.images, self.latent_posterior, name='Q')
        self.q.compile(
            loss=mutual_information,
            optimizer=Adam(lr=2e-4, beta_1=0.5, decay=2e-7))

        self.discriminator.trainable = False
        self.q.trainable = False
        self.infogan = Model(
            [
                self.noise,
                self.latent_prior,
            ], [
                self.discriminator(self.fake_images),
                self.q(self.fake_images),
            ],
            name='InfoGAN')
        self.infogan.compile(
            loss=['binary_crossentropy', mutual_information],
            optimizer=Adam(lr=2e-4, beta_1=0.5, decay=1e-7))
        # tensors = self._define_generator()
        # self.noise, self.latent_prior, self.fake_images = tensors
        # self.images, logits = self._define_discriminator()
        #
        # with K.name_scope('Q'):
        #     self.latent_posterior = Dense(
        #         self.latent_size, activation='softmax')(logits)
        # with K.name_scope('P'):
        #     self.probability = Dense(1, activation='sigmoid')(logits)
        #
        # self.generator = Model(
        #     [self.noise, self.latent_prior], self.fake_images, name='G')
        #
        # self.labels = Input(batch_shape=[None])
        # self.discriminator = Model(self.images, self.probability, name='D')
        # d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
        # print(d_vars)
        #
        # with tf.control_dependencies([tf.assert_positive(self.probability)]):
        #     self.discriminator_loss = losses.binary_crossentropy(
        #         self.labels, self.probability)
        #     self.d_opt = tf.train.AdamOptimizer(
        #         5e-4, beta1=0.5).minimize(
        #             self.discriminator_loss,
        #             var_list=d_vars + tf.get_collection(
        #                 tf.GraphKeys.TRAINABLE_VARIABLES, scope='P'))
        #
        # print(d_vars + tf.get_collection(
        #     tf.GraphKeys.TRAINABLE_VARIABLES, scope='P'))
        # print(d_vars + tf.get_collection(
        #     tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q'))
        # self.encoder = Model(self.images, self.latent_posterior, name='Q')
        # self.encoder_loss = losses.mutual_information(self.latent_prior,
        #                                               self.latent_posterior)
        # self.e_opt = tf.train.AdamOptimizer(
        #     2e-4, beta1=0.5).minimize(
        #         self.encoder_loss,
        #         var_list=d_vars + tf.get_collection(
        #             tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q'))
        #
        # print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G'))
        # self.infogan = Model(
        #     inputs=[self.noise, self.latent_prior],
        #     outputs=[
        #         self.discriminator(self.fake_images),
        #         self.encoder(self.fake_images)
        #     ],
        #     name='InfoGAN')
        # bce = -K.mean(K.log(self.infogan.outputs[0]))
        # mi = losses.mutual_information(self.latent_prior,
        #                                self.infogan.outputs[1])
        # self.infogan_loss = bce + mi
        # self.i_opt = tf.train.AdamOptimizer(
        #     2e-4, beta1=0.5).minimize(
        #         self.infogan_loss,
        #         var_list=tf.get_collection(
        #             tf.GraphKeys.TRAINABLE_VARIABLES, scope='G'))
        #
        # return dict(
        #     G=self.infogan_loss,
        #     Q=self.encoder_loss,
        #     D=self.discriminator_loss)

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
        images = self.generator.predict_on_batch([noise, latent_samples])
        # images = self.session.run(
        #     self.generator.output,
        #     feed_dict={
        #         self.latent_prior: latent_samples,
        #         self.noise: noise,
        #         K.learning_phase(): 0,
        #     })

        # Go from [-1, +1] scale back to [0, 1]
        return (images + 1) / 2

    def train_on_batch(self, real_images, with_summary=False):
        real_images = (real_images * 2) - 1
        batch_size = len(real_images)
        noise = self._sample_noise(batch_size)
        latent_code = self._sample_priors(batch_size)
        generated_images = self.generator.predict([noise, latent_code])
        assert len(generated_images) == len(real_images)
        all_images = np.concatenate([generated_images, real_images], axis=0)
        all_images += np.random.normal(0, 0.1, all_images.shape)

        labels = np.zeros(len(all_images))
        labels[batch_size:] = 1
        d_loss = self.discriminator.train_on_batch(all_images, labels)

        q_loss = self.q.train_on_batch(generated_images, latent_code)

        labels = np.ones(batch_size)
        noise = self._sample_noise(batch_size)
        latent_code = self._sample_priors(batch_size)
        g_loss, _, _ = self.infogan.train_on_batch([noise, latent_code],
                                                   [labels, latent_code])

        return dict(D=d_loss, G=g_loss, Q=q_loss)
        # batch_size = len(real_images)
        # real_images = (real_images * 2) - 1
        #
        # discriminator_loss = self._train_discriminator(real_images)
        # encoder_loss = self._train_encoder(batch_size)
        #
        # noise = self._sample_noise(batch_size)
        # latent = self._sample_priors(batch_size)
        # fetches = [self.i_opt, self.infogan_loss]
        # if with_summary:
        #     fetches.append(self.summary)
        #
        # # L_G = -D(G(z, c))
        # results = self.session.run(
        #     fetches,
        #     feed_dict={
        #         self.latent_prior: latent,
        #         self.noise: noise,
        #         K.learning_phase(): 1,
        #     })
        #
        # losses = dict(D=discriminator_loss, G=results[1], Q=encoder_loss)
        # return (losses, results[2]) if with_summary else losses

    def _add_summaries(self):
        super(InfoGAN, self)._add_summaries()
        # tf.summary.histogram('noise', self.noise)
        # tf.summary.histogram('latent_prior', self.latent_prior)
        # tf.summary.histogram('probability', self.infogan.outputs[0])
        # tf.summary.histogram('latent_posterior', self.infogan.outputs[1])
        # tf.summary.image('generated_images', self.fake_images, max_outputs=4)

    def _define_generator(self):
        with K.name_scope('G'):
            z = Input(shape=[self.noise_size])
            c = Input(shape=[self.latent_size])
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
        with K.name_scope('D'):
            x = Input(shape=self.image_shape)
            D = x
            for filters, stride in zip(self.discriminator_filters,
                                       self.discriminator_strides):
                D = Conv2D(
                    filters, (5, 5), strides=(stride, stride),
                    padding='same')(D)
                D = LeakyReLU(alpha=0.2)(D)
            D = Flatten()(D)

        with tf.control_dependencies([tf.assert_positive(D, [D])]):
            return x, D

    def _train_discriminator(self, real_images):
        latent = self._sample_priors(len(real_images))
        assert latent.shape == real_images.shape[:1] + (self.latent_size, )
        fake_images = self.generate(latent)
        labels = _get_labels(fake_images, real_images)
        assert labels.shape == (2 * len(real_images), )
        images = np.concatenate([fake_images, real_images], axis=0)
        # github.com/soumith/ganhacks#13-add-noise-to-inputs-decay-over-time
        images += np.random.normal(0, 0.1, images.shape)

        # L_D = -D(x) -D(G(z, c))
        _, discriminator_loss = self.session.run(
            [self.d_opt, self.discriminator_loss],
            feed_dict={
                self.images: images,
                self.labels: labels,
                K.learning_phase(): 1,
            })

        return discriminator_loss

    def _train_encoder(self, batch_size):
        latent = self._sample_priors(batch_size)
        fake_images = self.generate(latent)

        # I(c; G(z, c))
        _, encoder_loss = self.session.run(
            [self.e_opt, self.encoder_loss],
            feed_dict={
                self.images: fake_images,
                self.latent_prior: latent,
                K.learning_phase(): 1,
            })

        return encoder_loss

    def _add_optimizer(self, learning, loss):
        return 1.0, 2.0

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

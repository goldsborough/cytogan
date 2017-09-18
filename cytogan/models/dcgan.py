import collections

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Flatten, Input, LeakyReLU, Reshape, UpSampling2D,
                          Lambda)
from keras.models import Model

from cytogan.metrics import losses
from cytogan.models import model

Hyper = collections.namedtuple('Hyper', [
    'image_shape',
    'generator_filters',
    'discriminator_filters',
    'generator_strides',
    'discriminator_strides',
    'latent_size',
    'noise_size',
    'initial_shape',
])


class DCGAN(model.Model):
    def __init__(self, hyper, learning, session):
        assert len(hyper.image_shape) == 3
        # Copy all fields from hyper to self.
        for index, field in enumerate(hyper._fields):
            setattr(self, field, hyper[index])

        self.number_of_channels = hyper.image_shape[-1]

        self.images = None  # x
        self.labels = None  # 0/1

        self.batch_size = None
        self.noise = None  # z
        self.probability = None  # D(x)

        self.generator = None  # G(z, c)
        self.discriminator = None  # D(x)
        self.encoder = None
        self.gan = None  # D(G(z, c))

        super(DCGAN, self).__init__(learning, session)

    def _define_graph(self):
        with K.name_scope('G'):
            self.batch_size = Input(batch_shape=[1], name='batch_size')

            def sample_noise(batch_size):
                return tf.random_normal(shape=[
                    tf.squeeze(tf.cast(batch_size, tf.int32)), self.noise_size
                ])

            self.noise = Lambda(sample_noise)(self.batch_size)
            self.fake_images = self._define_generator(self.noise)

        self.images, logits = self._define_discriminator()

        self.latent = Dense(self.latent_size)(logits)
        self.probability = Dense(
            1, activation='sigmoid', name='D_final')(self.latent)

        self.generator = Model(self.batch_size, self.fake_images, name='G')
        self.encoder = Model(self.images, self.latent, name='E')

        self.loss = {}
        self.labels = Input(batch_shape=[None], name='labels')
        with K.name_scope('noisy_labels'):
            noisy_labels = self.labels * tf.random_uniform(
                tf.shape(self.labels), 0.8, 1.0)

        self.discriminator = Model(self.images, self.probability, name='D')
        with K.name_scope('D_loss'):
            self.loss['D'] = losses.binary_crossentropy(
                noisy_labels, self.discriminator.output)

        self.gan = Model(
            self.batch_size,
            self.discriminator(self.fake_images),
            name='DCGAN')
        with K.name_scope('G_loss'):
            self.loss['G'] = losses.binary_crossentropy(
                K.ones_like(self.gan.outputs[0]), self.gan.outputs[0])

    def encode(self, images):
        return self.encoder.predict_on_batch(np.array(images))

    def generate(self, latent_samples, rescale=True):
        feed_dict = {K.learning_phase(): 0}
        if isinstance(latent_samples, int):
            feed_dict[self.batch_size] = [latent_samples]
        else:
            feed_dict[self.noise] = latent_samples
        images = self.session.run(self.fake_images, feed_dict)
        # Go from [-1, +1] scale back to [0, 1]
        return (images + 1) / 2 if rescale else images

    def train_on_batch(self, real_images, with_summary=False):
        real_images = (real_images * 2) - 1
        fake_images = self.generate(len(real_images), rescale=False)
        assert fake_images.shape[1:] == real_images.shape[1:]

        d_loss = self._train_discriminator(fake_images, real_images)
        g_tensors = self._train_generator(len(real_images), with_summary)

        losses = dict(D=d_loss, G=g_tensors[0])
        return (losses, g_tensors[1]) if with_summary else losses

    @property
    def learning_rate(self):
        learning_rates = {}
        for key, lr in self._learning_rate.items():
            if isinstance(lr, tf.Tensor):
                lr = lr.eval(session=self.session)
            learning_rates[key] = lr
        return learning_rates

    def _add_summaries(self):
        super(DCGAN, self)._add_summaries()
        tf.summary.histogram('noise', self.noise)
        tf.summary.scalar('G_loss', self.loss['G'])
        tf.summary.image('generated_images', self.fake_images, max_outputs=8)

        fake_probability, real_probability = tf.split(self.gan.outputs[0], 2)
        tf.summary.histogram('fake_probability', fake_probability)
        tf.summary.histogram('real_probability', real_probability)

    def _define_generator(self, input_tensor):
        first_filter = self.generator_filters[0]
        G = Dense(np.prod(self.initial_shape) * first_filter)(input_tensor)
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

        G = Conv2D(self.number_of_channels, (5, 5), padding='same')(G)
        G = Activation('tanh')(G)
        assert G.shape[1:] == self.image_shape, G.shape

        return G

    def _define_discriminator(self):
        x = Input(shape=self.image_shape)

        # github.com/soumith/ganhacks#13-add-noise-to-inputs-decay-over-time
        def noisy(images):
            return images + tf.random_normal(
                tf.shape(images), mean=0.0, stddev=0.1)

        D = Lambda(noisy)(x)
        for filters, stride in zip(self.discriminator_filters,
                                   self.discriminator_strides):
            D = Conv2D(
                filters, (5, 5), strides=(stride, stride), padding='same')(D)
            D = LeakyReLU(alpha=0.2)(D)
        D = Flatten()(D)

        return x, D

    def _train_discriminator(self, fake_images, real_images):
        labels = np.concatenate(
            [np.zeros(len(fake_images)),
             np.ones(len(real_images))], axis=0)
        images = np.concatenate([fake_images, real_images], axis=0)

        # L_D = -D(x) -D(G(z, c))
        _, discriminator_loss = self.session.run(
            [self.optimizer['D'], self.loss['D']],
            feed_dict={
                self.images: images,
                self.labels: labels,
                K.learning_phase(): 1,
            })

        return discriminator_loss

    def _train_generator(self, batch_size, with_summary):
        fetches = [self.optimizer['G'], self.loss['G']]
        if with_summary:
            fetches.append(self.summary)

        results = self.session.run(
            fetches,
            feed_dict={
                self.batch_size: [batch_size],
                K.learning_phase(): 0,
            })

        return results[1:]

    def _add_optimizer(self, learning):
        self.optimizer = {}
        self._learning_rate = {}
        initial_learning_rate = learning.rate
        if isinstance(initial_learning_rate, float):
            initial_learning_rate = [initial_learning_rate] * 2

        with K.name_scope('D_opt'):
            self._learning_rate['D'] = self._get_learning_rate_tensor(
                initial_learning_rate[0], learning.decay,
                learning.steps_per_decay)
            self.optimizer['D'] = tf.train.AdamOptimizer(
                self._learning_rate['D'], beta1=0.5).minimize(
                    self.loss['D'],
                    var_list=self.discriminator.trainable_weights)

        with K.name_scope('G_opt'):
            self._learning_rate['G'] = self._get_learning_rate_tensor(
                initial_learning_rate[1], learning.decay,
                learning.steps_per_decay)
            self.optimizer['G'] = tf.train.AdamOptimizer(
                self._learning_rate['G'], beta1=0.5).minimize(
                    self.loss['G'],
                    var_list=self.generator.trainable_weights,
                    global_step=self.global_step)

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

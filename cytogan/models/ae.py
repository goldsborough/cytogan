import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten, Input, Reshape
from keras.models import Model
import keras.backend as K

from cytogan.metrics import losses
from cytogan.models import model

import collections

Hyper = collections.namedtuple('Hyper', 'image_shape, latent_size')


class AE(model.Model):
    def __init__(self, hyper, learning, session):
        assert len(hyper.image_shape) == 3
        self.image_shape = hyper.image_shape
        self.flat_image_shape = np.prod(hyper.image_shape)
        self.number_of_channels = hyper.image_shape[-1]
        self.latent_size = hyper.latent_size

        self.original_images = None
        self.reconstructed_images = None
        self.latent = None
        self.encoder = None

        super(AE, self).__init__(learning, session)

    def _define_graph(self):
        self.original_images = Input(shape=self.image_shape)
        flat_input = Flatten()(self.original_images)
        self.latent = Dense(self.latent_size, activation='relu')(flat_input)
        decoded = Dense(
            self.flat_image_shape, activation='sigmoid')(self.latent)
        self.reconstructed_images = Reshape(self.image_shape)(decoded)

        loss = K.mean(losses.reconstruction_loss(flat_input, decoded))

        self.model = Model(self.original_images, self.reconstructed_images)
        self.encoder = Model(self.original_images, self.latent)

        return loss

    def train_on_batch(self, batch, with_summary=False):
        fetches = [self.optimizer, self.loss]
        if with_summary is not None:
            fetches.append(self.summary)
        outputs = self.session.run(
            fetches, feed_dict={self.original_images: batch})
        if with_summary:
            return outputs[1:]
        return outputs[1]

    def encode(self, images):
        return self.session.run(
            self.encoder.output, feed_dict={self.original_images: images})

    def reconstruct(self, images):
        return self.session.run(
            self.model.output, feed_dict={self.original_images: images})

    def _add_summaries(self):
        super(AE, self)._add_summaries()
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('learning_rate', self._learning_rate)
        tf.summary.histogram('latent', self.latent)

    def _add_optimizer(self, learning, loss):
        learning_rate = self._get_learning_rate_tensor(
            learning.rate, learning.decay, learning.steps_per_decay)
        loss = tf.check_numerics(loss, self.name)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        optimization = optimizer.minimize(loss, self.global_step)

        return learning_rate, optimization

    def __repr__(self):
        lines = [self.name]
        try:
            # >= Keras 2.0.6
            self.model.summary(print_fn=lines.append)
        except TypeError:
            lines = [layer.name for layer in self.model.layers]
        return '\n'.join(map(str, lines))

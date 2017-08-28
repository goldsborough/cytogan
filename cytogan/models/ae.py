import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten, Input, Reshape
from keras.models import Model
import keras.backend as K

from cytogan.metrics import losses
import cytogan.models.model

import collections

Hyper = collections.namedtuple('Hyper', 'image_shape, latent_size')


class AE(cytogan.models.model.Model):
    def __init__(self, hyper, learning, session):
        assert len(hyper.image_shape) == 3
        self.image_shape = hyper.image_shape
        self.flat_image_shape = np.prod(hyper.image_shape)
        self.latent_size = hyper.latent_size

        self.original_images = None  # = self.input
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

        model = Model(self.original_images, self.reconstructed_images)
        self.encoder = Model(self.original_images, self.latent)

        return self.original_images, loss, model

    def encode(self, images):
        return self.session.run(
            self.encoder.output, feed_dict={self.original_images: images})

    def reconstruct(self, images):
        return self.session.run(
            self.model.output, feed_dict={self.original_images: images})

    def _add_summaries(self):
        tf.summary.histogram('latent', self.latent)
        super(AE, self)._add_summaries()

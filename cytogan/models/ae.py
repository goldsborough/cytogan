import keras.backend as K
import keras.optimizers
import numpy as np
from keras.layers import Dense, Flatten, Input, Reshape
from keras.models import Model


class AE(object):
    def __init__(self, image_shape, latent_size):
        assert len(image_shape) == 3
        self.image_shape = image_shape
        self.flat_image_shape = np.prod(image_shape)
        self.latent_size = latent_size
        self.encoder = None
        self.model = None
        self.optimizer = None

    def compile(self, learning_rate, decay_learning_rate_after,
                learning_rate_decay):
        original_images = Input(shape=self.image_shape)
        flat_images = Flatten()(original_images)
        latent = Dense(self.latent_size, activation='relu')(flat_images)
        decoded = Dense(self.flat_image_shape, activation='sigmoid')(latent)
        reconstruction = Reshape(self.image_shape)(decoded)

        self.encoder = Model(original_images, latent)
        self.model = Model(original_images, reconstruction)

        self._attach_optimizer(learning_rate, decay_learning_rate_after,
                               learning_rate_decay)

    @property
    def is_ready(self):
        return self.model is not None

    @property
    def learning_rate(self):
        assert hasattr(self, 'optimizer'), 'must call compile() first'
        exp = (1. / (1. + self.optimizer.decay * self.optimizer.iterations))
        return K.eval(self.optimizer.lr * exp)

    def train_on_batch(self, images):
        assert self.is_ready
        return self.model.train_on_batch(images, images)

    def encode(self, images):
        assert self.is_ready
        return self.encoder.predict(images)

    def reconstruct(self, images):
        assert self.is_ready
        return self.model.predict(images)

    def _attach_optimizer(self,
                          learning_rate,
                          decay_learning_rate_after,
                          learning_rate_decay,
                          loss='binary_crossentropy'):
        # TF treats the decay as a factor every N steps, while for Keras it's d
        # in lr^(1 / (1 + d * iterations)).
        self.optimizer = keras.optimizers.Adam(
            lr=learning_rate, decay=1 - learning_rate_decay)
        self.model.compile(loss=loss, optimizer=self.optimizer)

    def __repr__(self):
        assert self.is_ready
        lines = []
        try:
            # >= Keras 2.0.6
            self.model.summary(print_fn=lambda l: lines.append(l))
        except TypeError:
            lines = [layer.name for layer in self.model.layers]
        return '\n'.join(map(str, lines))

import keras.backend as K
import keras.optimizers
import numpy as np
from keras.layers import Dense, Flatten, Input, Reshape
from keras.models import Model


class AE(object):
    def __init__(self, image_shape, latent_size=32):
        original_images = Input(shape=image_shape)
        flat_images = Flatten()(original_images)
        latent = Dense(latent_size, activation='relu')(flat_images)
        decoded = Dense(np.prod(image_shape), activation='sigmoid')(latent)
        reconstruction = Reshape(image_shape)(decoded)

        self.latent_model = Model(original_images, latent)
        self.reconstruction_model = Model(original_images, reconstruction)

    @property
    def learning_rate(self):
        exp = (1. / (1. + self.optimizer.decay * self.optimizer.iterations))
        return K.eval(self.optimizer.lr * exp)

    def prepare(self, learning_rate, decay_learning_rate_after,
                learning_rate_decay):
        # TF treats the decay as a factor every N steps, while for Keras it's d
        # in lr^(1 / (1 + d * iterations)).
        self.optimizer = keras.optimizers.Adam(
            lr=learning_rate, decay=1 - learning_rate_decay)
        self.reconstruction_model.compile(
            loss='binary_crossentropy', optimizer=self.optimizer)

    def train_on_batch(self, images):
        return self.reconstruction_model.train_on_batch(images, images)

    def generate(self, images):
        latent_vectors = self.latent_model.predict(images)
        reconstructions = self.reconstruction_model.predict(images)
        return latent_vectors, reconstructions

import keras.backend as K
import keras.losses
import numpy as np
from keras.layers import (Conv2D, Dense, Flatten, Input, MaxPooling2D, Reshape,
                          UpSampling2D)
from keras.models import Model

from cytogan.models import ae
from cytogan.metrics import losses


def build_encoder(original_images, filter_sizes):
    assert len(filter_sizes) > 0
    conv = original_images
    for filter_size in filter_sizes:
        conv = Conv2D(
            filter_size, kernel_size=(3, 3), activation='relu',
            padding='same')(conv)
        conv = MaxPooling2D((2, 2), padding='same')(conv)

    return conv, Flatten()(conv)


def build_decoder(last_encoder_layer, latent, filter_sizes):
    first_shape = list(map(int, last_encoder_layer.shape[1:]))
    deconv_flat = Dense(np.prod(first_shape))(latent)
    deconv = Reshape(first_shape)(deconv_flat)
    deconv = UpSampling2D((2, 2))(deconv)
    # Go through encoder layers in reverse order and skip the first layer.
    for filter_size in filter_sizes[:0:-1]:
        deconv = Conv2D(
            filter_size, kernel_size=(3, 3), activation='relu',
            padding='same')(deconv)

    return deconv


class ConvAE(ae.AE):
    def __init__(self, image_shape, filter_sizes, latent_size):
        super(ConvAE, self).__init__(image_shape, latent_size)
        self.filter_sizes = filter_sizes

    def compile(self, learning_rate, decay_learning_rate_after,
                learning_rate_decay):
        self.original_images = Input(shape=self.image_shape)

        conv, conv_flat = build_encoder(self.original_images,
                                        self.filter_sizes)
        self.latent = Dense(self.latent_size, activation='relu')(conv_flat)
        deconv = build_decoder(conv, self.latent, self.filter_sizes)

        self.reconstructed_images = Conv2D(
            self.image_shape[2], (3, 3), activation='sigmoid',
            padding='same')(deconv)

        self.loss = losses.reconstruction_loss(self.original_images,
                                               self.reconstructed_images)

        self.encoder = Model(self.original_images, self.latent)
        self.model = Model(self.original_images, self.reconstructed_images)

        self.optimize = self._add_optimization_target(
            learning_rate, decay_learning_rate_after, learning_rate_decay)
        self.summary = self._add_summary()

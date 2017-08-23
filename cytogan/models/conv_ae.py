import numpy as np
from keras.layers import (Conv2D, Dense, Flatten, Input, MaxPooling2D, Reshape,
                          UpSampling2D)
from keras.models import Model

import cytogan.models.ae


def build_encoder(original_images, filter_sizes):
    previous_layer = original_images
    for filter_size in filter_sizes:
        conv = Conv2D(
            filter_size, kernel_size=(3, 3), activation='relu',
            padding='same')(previous_layer)
        conv = MaxPooling2D((2, 2), padding='same')(conv)
        previous_layer = conv

    return conv, Flatten()(conv)


def build_decoder(last_encoder_layer, latent, filter_sizes):
    first_shape = list(map(int, last_encoder_layer.shape[1:]))
    deconv_flat = Dense(np.prod(first_shape))(latent)
    previous_layer = Reshape(first_shape)(deconv_flat)
    for filter_size in filter_sizes[::-1]:
        deconv = Conv2D(
            filter_size, kernel_size=(3, 3), activation='relu',
            padding='same')(previous_layer)
        deconv = UpSampling2D((2, 2))(deconv)
        previous_layer = deconv

    return deconv


class ConvAE(cytogan.models.ae.AE):
    def __init__(self, image_shape, filter_sizes, latent_size):
        super(ConvAE, self).__init__(image_shape, latent_size)
        self.filter_sizes = filter_sizes

    def compile(self, learning_rate, decay_learning_rate_after,
                learning_rate_decay):
        original_images = Input(shape=self.image_shape)

        conv, conv_flat = build_encoder(original_images, self.filter_sizes)
        latent = Dense(self.latent_size, activation='relu')(conv_flat)
        deconv = build_decoder(conv, latent, self.filter_sizes)

        reconstruction = Conv2D(
            self.image_shape[2], (3, 3), activation='sigmoid',
            padding='same')(deconv)

        self.encoder = Model(original_images, latent)
        self.model = Model(original_images, reconstruction)

        self._attach_optimizer(learning_rate, decay_learning_rate_after,
                               learning_rate_decay)

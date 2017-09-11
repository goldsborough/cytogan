import collections

import keras.backend as K
import numpy as np
from keras.layers import (Conv2D, Dense, Flatten, Input, MaxPooling2D, Reshape,
                          UpSampling2D)
from keras.models import Model

from cytogan.metrics import losses
from cytogan.models import ae

Hyper = collections.namedtuple('Hyper',
                               'image_shape, filter_sizes, latent_size')


def build_encoder(original_images, filter_sizes):
    assert len(filter_sizes) > 0
    conv = original_images
    with K.name_scope('encoder'):
        for filter_size in filter_sizes:
            conv = Conv2D(
                filter_size,
                kernel_size=(3, 3),
                activation='relu',
                padding='same')(conv)
            conv = MaxPooling2D((2, 2), padding='same')(conv)
        flat = Flatten()(conv)

    return conv, flat


def build_decoder(last_encoder_layer, latent, filter_sizes):
    first_shape = list(map(int, last_encoder_layer.shape[1:]))
    with K.name_scope('decoder'):
        deconv_flat = Dense(np.prod(first_shape))(latent)
        deconv = Reshape(first_shape)(deconv_flat)
        deconv = UpSampling2D((2, 2))(deconv)
        # Go through encoder layers in reverse order and skip the last layer.
        # [-2::-1] means start at the second to last (inclusive) and go to 0 in
        # steps of -1.
        for filter_size in filter_sizes[-2::-1]:
            deconv = Conv2D(
                filter_size,
                kernel_size=(3, 3),
                activation='relu',
                padding='same')(deconv)
            deconv = UpSampling2D((2, 2))(deconv)

    return deconv


class ConvAE(ae.AE):
    def __init__(self, hyper, learning, session):
        self.filter_sizes = hyper.filter_sizes
        super(ConvAE, self).__init__(hyper, learning, session)

    def _define_graph(self):
        self.original_images = Input(shape=self.image_shape)

        conv, conv_flat = build_encoder(self.original_images,
                                        self.filter_sizes)
        self.latent = Dense(self.latent_size, activation='relu')(conv_flat)
        deconv = build_decoder(conv, self.latent, self.filter_sizes)

        self.reconstructed_images = Conv2D(
            self.image_shape[2], (3, 3), activation='sigmoid',
            padding='same')(deconv)

        batch_loss = losses.reconstruction_loss(self.original_images,
                                                self.reconstructed_images)
        loss = K.mean(batch_loss)

        self.encoder = Model(self.original_images, self.latent)
        model = Model(self.original_images, self.reconstructed_images)

        return self.original_images, loss, model

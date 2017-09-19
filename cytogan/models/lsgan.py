import keras.backend as K
import keras.losses
from keras.layers import Dense

from cytogan.models import dcgan

Hyper = dcgan.Hyper


class LSGAN(dcgan.DCGAN):
    def __init__(self, hyper, learning, session):
        super(LSGAN, self).__init__(hyper, learning, session)

    def _define_discriminator_loss(self, labels, probability):
        noisy_labels = dcgan.smooth_labels(labels)
        with K.name_scope('D_loss'):
            probability = K.squeeze(probability, 1)
            return keras.losses.mean_squared_error(noisy_labels, probability)

    def _define_generator_loss(self, probability):
        with K.name_scope('G_loss'):
            probability = K.squeeze(probability, 1)
            ones = K.ones_like(probability)
            return keras.losses.mean_squared_error(ones, probability)

    def _define_final_discriminator_layer(self, latent):
        # No activation for LSGAN
        return Dense(1, name='D_final')(latent)

import keras.backend as K
from keras.layers import Dense

from cytogan.models import gan, dcgan
from cytogan.metrics import losses

Hyper = dcgan.Hyper


class LSGAN(dcgan.DCGAN):
    def __init__(self, hyper, learning, session):
        super(LSGAN, self).__init__(hyper, learning, session)

    def _define_discriminator_loss(self, labels, probability):
        noisy_labels = gan.smooth_labels(labels)
        with K.name_scope('D_loss'):
            return losses.mean_squared_error(noisy_labels, probability)

    def _define_generator_loss(self, probability):
        with K.name_scope('G_loss'):
            ones = K.ones_like(probability)
            return losses.mean_squared_error(ones, probability)

    def _define_final_discriminator_layer(self, latent):
        # No activation for LSGAN
        return Dense(1, name='D_final')(latent)

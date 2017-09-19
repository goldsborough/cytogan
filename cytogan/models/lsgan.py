import keras.backend as K
from keras.layers import Dense
import keras.losses

from cytogan.models import dcgan

Hyper = dcgan.Hyper


class LSGAN(dcgan.DCGAN):
    def __init__(self, hyper, learning, session):
        super(LSGAN, self).__init__(hyper, learning, session)

    def _define_discriminator_loss(self, labels):
        noisy_labels = dcgan.smooth_labels(labels)
        with K.name_scope('D_loss'):
            return keras.losses.mean_squared_error(noisy_labels, self.d_final)

    def _define_generator_loss(self):
        with K.name_scope('G_loss'):
            probability = self.gan.outputs[0]
            ones = K.ones_like(probability)
            return keras.losses.mean_squared_error(ones, probability)

    def _define_final_discriminator_layer(self, latent):
        # No activation for LSGAN
        return Dense(1, name='D_final')(latent)

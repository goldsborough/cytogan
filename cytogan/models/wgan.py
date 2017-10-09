import keras.backend as K
import tensorflow as tf
from keras.layers import Dense

from cytogan.models import dcgan

Hyper = dcgan.Hyper


class WGAN(dcgan.DCGAN):
    def __init__(self, hyper, learning, session):
        super(WGAN, self).__init__(hyper, learning, session)

    def _define_discriminator_loss(self, _, logits):
        with K.name_scope('D_loss'):
            generated_logits, real_logits = tf.split(logits, 2)

            with K.name_scope('wasserstein_distance'):
                loss = K.mean(generated_logits) - K.mean(real_logits)

            with K.name_scope('gradient_penalty'):
                batch_size = tf.cast(tf.squeeze(self.batch_size), tf.int32)
                available = tf.shape(self.images)[0] - batch_size
                generated_images = self.images[:available]
                real_images = self.images[-available:]
                epsilon = tf.random_uniform(shape=K.shape(generated_images))
                mix = epsilon * real_images + (1 - epsilon) * generated_images
                inputs = [mix]
                if self.is_conditional:
                    inputs.append(self.conditional['D'])
                gradients = K.gradients(self.discriminator(inputs), mix)[0]
                slopes = K.sqrt(K.sum(K.square(gradients), axis=[1, 2, 3]))
                gradient_penalty = 10 * K.mean(K.square(slopes - 1), axis=0)

            return loss + gradient_penalty

    def _define_generator_loss(self, logits):
        with K.name_scope('G_loss'):
            return -K.mean(logits)

    def _define_final_discriminator_layer(self, latent):
        # No activation for WGAN
        return Dense(1, name='D_final')(latent)

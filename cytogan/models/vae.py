import keras.backend as K
import keras.losses
import keras.optimizers
from keras.layers import Conv2D, Dense, Input, Lambda
from keras.models import Model
import tensorflow as tf

import cytogan.models.ae
from cytogan.models import conv_ae


def _reuse_decoder_layers(model, latent_size):
    # Find the index of the latent layer, then reuse all layers after that.
    latent_index = [l.name for l in model.layers].index('latent')
    decoder_input = Input(shape=[latent_size])
    decoder_layer = decoder_input
    for deconv_layer in model.layers[latent_index + 1:]:
        decoder_layer = deconv_layer(decoder_layer)
    return Model(decoder_input, decoder_layer)


class VAE(cytogan.models.ae.AE):
    def __init__(self, image_shape, filter_sizes, latent_size):
        super(VAE, self).__init__(image_shape, latent_size)
        self.filter_sizes = filter_sizes
        self.decoder = None

        # Tensor handles for the loss function.
        self.mean = None
        self.sigma = None
        self.log_sigma = None

    def compile(self, learning_rate, decay_learning_rate_after,
                learning_rate_decay):
        original_images = Input(shape=self.image_shape)
        conv, conv_flat = conv_ae.build_encoder(original_images,
                                                self.filter_sizes)

        self.mean = Dense(self.latent_size)(conv_flat)
        log_sigma = Dense(self.latent_size)(conv_flat)
        self.sigma = K.exp(log_sigma)

        latent = Lambda(
            self._sample_latent, name='latent')([self.mean, log_sigma])
        deconv = conv_ae.build_decoder(conv, latent, self.filter_sizes)

        reconstruction = Conv2D(
            self.image_shape[-1], (3, 3), activation='sigmoid',
            padding='same')(deconv)
        assert reconstruction.shape[1:] == self.image_shape

        self.encoder = Model(original_images, latent)
        self.model = Model(original_images, reconstruction)
        self.decoder = _reuse_decoder_layers(self.model, self.latent_size)

        self._attach_optimizer(
            learning_rate,
            decay_learning_rate_after,
            learning_rate_decay,
            loss=self._loss)

    def decode(self, samples):
        assert self.is_ready
        return self.decoder.predict(samples)

    def _sample_latent(self, tensors):
        mean, log_sigma = tensors
        noise = tf.random_normal(
            shape=[tf.shape(mean)[0], self.latent_size], stddev=0.1)
        return mean + K.exp(log_sigma) * noise

    def _loss(self, original_images, reconstructed_images):
        # K.flatten() doesn't preserve the batch size here :(
        flat_shape = [-1, self.flat_image_shape]
        original_images_flat = K.reshape(original_images, flat_shape)
        reconstructed_images_flat = K.reshape(reconstructed_images, flat_shape)

        e = 1e-10  # numerical stability
        binary_cross_entropies = original_images_flat * tf.log(
            e + reconstructed_images_flat) + (
                1 - original_images_flat
            ) * tf.log(e + 1 - reconstructed_images_flat)
        reconstruction_loss = -tf.reduce_sum(binary_cross_entropies, axis=1)

        regularization_loss = -0.5 * tf.reduce_sum(
            1 + tf.log(e + tf.square(self.sigma)) - tf.square(self.mean) -
            tf.square(self.sigma),
            axis=1)

        return tf.reduce_mean(regularization_loss + reconstruction_loss)

        # reconstruction_loss = -tf.reduce_mean(binary_cross_entropies, axis=1)
        # # reconstruction_loss = keras.losses.binary_crossentropy(
        # #     original_images_flat, reconstructed_images_flat)
        # assert len(reconstruction_loss.shape) == 1
        #
        # # https://arxiv.org/abs/1312.6114, Appendix B
        # regularization_loss = -0.5 * K.sum(
        #     1 + K.log(1e-10 + K.square(self.sigma)) - K.square(self.mean) -
        #     K.square(self.sigma),
        #     axis=1)
        # assert len(regularization_loss.shape) == 1
        #
        # return K.mean(regularization_loss + reconstruction_loss)

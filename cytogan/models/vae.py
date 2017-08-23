from keras.layers import Conv2D, Dense, Input, Lambda
import keras.optimizers
from keras.models import Model
import keras.backend as K
import keras.metrics

import cytogan.models.ae
from cytogan.models import conv_ae


def reuse_decoder_layers(model, latent, filter_sizes):
    # Find the index of the latent layer, then reuse all layers after that.
    latent_index = [l.name for l in model.layers].index('latent')
    decoder_input = Input(shape=[int(latent.shape[1])])
    decoder_layer = decoder_input
    for deconv_layer in model.layers[latent_index + 1:]:
        decoder_layer = deconv_layer(decoder_layer)
    return Model(decoder_input, decoder_layer)


class VAE(cytogan.models.ae.AE):
    def __init__(self, batch_size, image_shape, filter_sizes, latent_size):
        assert 2 <= len(image_shape) <= 3
        self.batch_size = batch_size
        self.latent_size = latent_size

        original_images = Input(shape=image_shape)
        conv, conv_flat = conv_ae.build_encoder(original_images, filter_sizes)

        self.mean = Dense(latent_size)(conv_flat)
        self.log_sigma = Dense(latent_size)(conv_flat)
        self.sigma = K.exp(self.log_sigma)

        latent = Lambda(
            self._sample_latent, name='latent')([self.mean, self.log_sigma])
        deconv = conv_ae.build_decoder(conv, latent, filter_sizes)

        reconstruction = Conv2D(
            image_shape[2], (3, 3), activation='sigmoid',
            padding='same')(deconv)

        self.encoder = Model(original_images, latent)
        self.model = Model(original_images, reconstruction)
        self.decoder = reuse_decoder_layers(self.model, latent, filter_sizes)

    def _sample_latent(self, tensors):
        mean, log_sigma = tensors
        noise = K.random_normal(
            shape=[self.batch_size, self.latent_size], stddev=0.1)
        return mean + K.exp(log_sigma) * noise

    def _loss(self, original_images, reconstructed_images):
        reconstruction_loss = keras.metrics.binary_crossentropy(
            original_images, reconstructed_images)

        regularization_loss = -0.5 * K.sum(
            1 + K.log(1e-10 + K.square(self.sigma)) - K.square(self.mean) -
            K.square(self.sigma),
            axis=1)

        return K.mean(regularization_loss + reconstruction_loss)

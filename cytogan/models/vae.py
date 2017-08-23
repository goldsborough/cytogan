from keras.layers import Conv2D, Dense, Input, Lambda
import keras.optimizers
from keras.models import Model
import keras.backend as K
import keras.metrics

import cytogan.models.ae
from cytogan.models import conv_ae


def reuse_decoder_layers(model, latent):
    # Find the index of the latent layer, then reuse all layers after that.
    latent_index = [l.name for l in model.layers].index('latent')
    decoder_input = Input(shape=[int(latent.shape[1])])
    decoder_layer = decoder_input
    for deconv_layer in model.layers[latent_index + 1:]:
        decoder_layer = deconv_layer(decoder_layer)
    return Model(decoder_input, decoder_layer)


class VAE(cytogan.models.ae.AE):
    def __init__(self, batch_size, image_shape, filter_sizes, latent_size):
        super(VAE, self).__init__(image_shape, latent_size)
        self.batch_size = batch_size
        self.filter_sizes = filter_sizes
        self.decoder = None

        # Tensor handles
        self.mean = None
        self.sigma = None
        self.log_sigma = None

    def compile(self, learning_rate, decay_learning_rate_after,
                learning_rate_decay):
        original_images = Input(shape=self.image_shape)
        conv, conv_flat = conv_ae.build_encoder(original_images,
                                                self.filter_sizes)

        self.mean = Dense(self.latent_size)(conv_flat)
        self.log_sigma = Dense(self.latent_size)(conv_flat)
        self.sigma = K.exp(self.log_sigma)

        latent = Lambda(
            self._sample_latent, name='latent')([self.mean, self.log_sigma])
        deconv = conv_ae.build_decoder(conv, latent, self.filter_sizes)

        reconstruction = Conv2D(
            self.image_shape[2], (3, 3), activation='sigmoid',
            padding='same')(deconv)

        self.encoder = Model(original_images, latent)
        self.model = Model(original_images, reconstruction)
        self.decoder = reuse_decoder_layers(self.model, latent)

        self._attach_optimizer(
            learning_rate,
            decay_learning_rate_after,
            learning_rate_decay,
            loss=self._loss)

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

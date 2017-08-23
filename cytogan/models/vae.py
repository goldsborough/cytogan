from keras.layers import Conv2D, Dense, Input, Lambda
from keras.layers.merge import Concatenate
from keras.models import Model
import keras.backend as K

import cytogan.models.ae
from cytogan.models import conv_ae


def reuse_decoder_layers(model, latent, filter_sizes):
    latent_index = [l.name for l in model.layers].index('latent')
    decoder_input = Input(shape=[int(latent.shape[1])])
    decoder_layer = decoder_input
    for deconv_layer in model.layers[latent_index + 1:]:
        decoder_layer = deconv_layer(decoder_layer)
    return Model(decoder_input, decoder_layer)


class VAE(cytogan.models.ae.AE):
    def __init__(self, batch_size, image_shape, filter_sizes, latent_size):
        assert 2 <= len(image_shape) <= 3
        original_images = Input(shape=image_shape)

        conv, conv_flat = conv_ae.build_encoder(original_images, filter_sizes)

        mean = Dense(latent_size)(conv_flat)
        log_sigma = Dense(latent_size)(conv_flat)

        def sample_latent(tensors):
            mean, log_sigma = tensors
            noise = K.random_normal(
                shape=[batch_size, latent_size], stddev=0.1)
            return mean + K.exp(log_sigma) * noise

        latent = Lambda(sample_latent, name='latent')([mean, log_sigma])
        deconv = conv_ae.build_decoder(conv, latent, filter_sizes)

        reconstruction = Conv2D(
            image_shape[2], (3, 3), activation='sigmoid',
            padding='same')(deconv)

        self.encoder = Model(original_images, latent)
        self.model = Model(original_images, reconstruction)
        self.decoder = reuse_decoder_layers(self.model, latent, filter_sizes)

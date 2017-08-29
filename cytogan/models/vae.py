import keras.backend as K
import tensorflow as tf
from keras.layers import Conv2D, Dense, Input, Lambda
from keras.models import Model

from cytogan.models import conv_ae
from cytogan.metrics import losses

Hyper = conv_ae.Hyper


def _reuse_decoder_layers(model, latent_size):
    # Find the index of the latent layer, then reuse all layers after that.
    latent_index = [l.name for l in model.layers].index('latent')
    decoder_input = Input(shape=[latent_size])
    decoder_layer = decoder_input
    for deconv_layer in model.layers[latent_index + 1:]:
        decoder_layer = deconv_layer(decoder_layer)
    return Model(decoder_input, decoder_layer)


class VAE(conv_ae.ConvAE):
    def __init__(self, hyper, learning, session):
        self.decoder = None
        # Tensor handles for the loss function.
        self.mean = None
        self.sigma = None
        self.log_sigma = None

        super(VAE, self).__init__(hyper, learning, session)

    def decode(self, samples):
        return self.session.run(
        self.decoder.output, feed_dict={self.decoder.input: samples})

    def _define_graph(self):
        self.original_images = Input(shape=self.image_shape)
        conv, conv_flat = conv_ae.build_encoder(self.original_images,
                                                self.filter_sizes)

        self.mean = Dense(self.latent_size)(conv_flat)
        log_sigma = Dense(self.latent_size)(conv_flat)
        self.sigma = K.exp(log_sigma)

        self.latent = Lambda(
            self._sample_latent, name='latent')([self.mean, log_sigma])
        deconv = conv_ae.build_decoder(conv, self.latent, self.filter_sizes)

        self.reconstructed_images = Conv2D(
            self.image_shape[-1], (3, 3), activation='sigmoid',
            padding='same')(deconv)
        assert self.reconstructed_images.shape[1:] == \
            self.image_shape, self.reconstructed_images.shape

        loss = self._add_loss(self.original_images, self.reconstructed_images)

        self.encoder = Model(self.original_images, self.latent)
        model = Model(self.original_images, self.reconstructed_images)
        self.decoder = _reuse_decoder_layers(model, self.latent_size)

        return self.original_images, loss, model

    def _sample_latent(self, tensors):
        mean, log_sigma = tensors
        noise = tf.random_normal(
            shape=[tf.shape(mean)[0], self.latent_size], stddev=0.1)
        return mean + K.exp(log_sigma) * noise

    def _add_loss(self, original_images, reconstructed_images):
        reconstruction_loss = losses.reconstruction_loss(
            original_images, reconstructed_images)

        # https://arxiv.org/abs/1312.6114, Appendix B
        regularization_loss = -0.5 * K.sum(
            1 + K.log(1e-10 + K.square(self.sigma)) - K.square(self.mean) -
            K.square(self.sigma),
            axis=1)
        assert len(regularization_loss.shape) == 1

        return K.mean(regularization_loss + reconstruction_loss)

    def _add_summaries(self):
        super(VAE, self)._add_summaries()
        tf.summary.histogram('latent_mean', self.mean)
        tf.summary.histogram('latent_stddev', self.sigma)

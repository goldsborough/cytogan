import keras.optimizers
from keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model


class VAE(object):
    def __init__(self, image_shape):
        assert 2 <= len(image_shape) <= 3
        original_images = Input(shape=image_shape)

        # input -> 16 -> 8 -> 8 -> latent
        conv = Conv2D(
            16, (3, 3), activation='relu', padding='same')(original_images)
        conv = MaxPooling2D((2, 2), padding='same')(conv)
        conv = Conv2D(8, (3, 3), activation='relu', padding='same')(conv)
        conv = MaxPooling2D((2, 2), padding='same')(conv)
        conv = Conv2D(8, (3, 3), activation='relu', padding='same')(conv)
        conv = MaxPooling2D((2, 2), padding='same')(conv)

        latent = Flatten()(conv)
        print(latent.shape)

        # latent -> 8 -> 8 -> 16 -> input
        deconv = Conv2D(8, (3, 3), activation='relu', padding='same')(conv)
        deconv = UpSampling2D((2, 2))(deconv)
        deconv = Conv2D(8, (3, 3), activation='relu', padding='same')(deconv)
        deconv = UpSampling2D((2, 2))(deconv)
        deconv = Conv2D(16, (3, 3), activation='relu')(deconv)
        deconv = UpSampling2D((2, 2))(deconv)
        reconstruction = Conv2D(
            image_shape[2], (3, 3), activation='sigmoid',
            padding='same')(deconv)

        self.latent_model = Model(original_images, latent)
        self.reconstruction_model = Model(original_images, reconstruction)

    def prepare(self, learning_rate, decay_learning_rate_after,
                learning_rate_decay):
        optimizer = keras.optimizers.Adam(
            lr=learning_rate, decay=learning_rate_decay)
        self.reconstruction_model.compile(
            loss='binary_crossentropy', optimizer=optimizer)

    def train_on_batch(self, images):
        return self.reconstruction_model.train_on_batch(images, images)

    def generate(self, images, session):
        latent_vectors = self.latent_model.predict(images)
        reconstructions = self.reconstruction_model.predict(images)
        return latent_vectors, reconstructions

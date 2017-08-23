from keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model

import cytogan.models.ae


class VAE(cytogan.models.ae.AE):
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

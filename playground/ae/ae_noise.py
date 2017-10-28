#!/usr/bin/env python3

import matplotlib.pyplot as plot
import numpy as np

from keras.models import Model
from keras.datasets import mnist
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(
    loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(
    loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Model

input_img = Input(shape=(28, 28, 1))

conv = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
conv = MaxPooling2D((2, 2), padding='same')(conv)
conv = Conv2D(32, (3, 3), activation='relu', padding='same')(conv)
encoded = MaxPooling2D((2, 2), padding='same')(conv)

deconv = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
deconv = UpSampling2D((2, 2))(deconv)
deconv = Conv2D(32, (3, 3), activation='relu', padding='same')(deconv)
deconv = UpSampling2D((2, 2))(deconv)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(deconv)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(
    x_train_noisy,
    x_train,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test_noisy, x_test))

x_test_denoised = autoencoder.predict(x_test_noisy)

n = 10
plot.figure(figsize=(30, 4))
for i in range(n):
    ax = plot.subplot(3, n, i + 1)
    plot.imshow(x_test_noisy[i].reshape(28, 28))
    plot.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plot.subplot(3, n, n + i + 1)
    plot.imshow(x_test_denoised[i].reshape(28, 28))
    plot.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plot.subplot(3, n, n + n + i + 1)
    plot.imshow(x_test[i].reshape(28, 28))
    plot.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plot.show()

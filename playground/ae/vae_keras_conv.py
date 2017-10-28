#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10, mnist
from keras.layers import (Conv2D, Dense, Flatten, Input, Lambda, MaxPool2D,
                          Reshape, UpSampling2D)
from keras.models import Model


def mnist_data():
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    return x_train, x_test, (28, 28, 1)


def cifar_data():
    (x_train, _), (x_test, _) = cifar10.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    return x_train, x_test, (32, 32, 3)


x_train, x_test, original_shape = cifar_data()

latent_size = 128
input_shape = x_train[0].shape
flat_input_shape = np.prod(input_shape)

input_images = Input(input_shape)

conv = Conv2D(32, (3, 3), activation='relu', padding='same')(input_images)
conv = MaxPool2D((2, 2), padding='same')(conv)
conv = Conv2D(32, (3, 3), activation='relu', padding='same')(conv)
conv = MaxPool2D((2, 2), padding='same')(conv)
conv = Conv2D(32, (3, 3), activation='relu', padding='same')(conv)
conv = MaxPool2D((2, 2), padding='same')(conv)
encoded = Flatten()(conv)

mean = Dense(latent_size)(encoded)
log_sigma = Dense(latent_size)(encoded)
sigma = tf.exp(log_sigma)


def latent_op(tensors):
    mean, log_sigma = tensors
    noise = tf.truncated_normal(
        shape=[tf.shape(mean)[0], latent_size], stddev=0.1)
    return mean + tf.exp(log_sigma) * noise


last_shape = list(map(int, conv.shape[1:]))

latent = Lambda(latent_op)([mean, log_sigma])
deconv = Dense(np.prod(last_shape))(latent)
deconv = Reshape(last_shape)(deconv)  # 4x4
deconv = UpSampling2D(2)(deconv)  # 8x8
deconv = Conv2D(32, (3, 3), activation='relu', padding='same')(deconv)
deconv = UpSampling2D((2, 2))(deconv)  # 16x16
deconv = Conv2D(32, (3, 3), activation='relu', padding='same')(deconv)  # 14x14
deconv = UpSampling2D((2, 2))(deconv)  # 28x28
decoded = Conv2D(
    input_shape[-1], (3, 3), activation='sigmoid', padding='same')(deconv)


def vae_loss(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, [-1, flat_input_shape])
    y_pred_flat = tf.reshape(y_pred, [-1, flat_input_shape])
    reconstruction_loss = tf.reduce_sum(
        tf.square(y_true_flat - y_pred_flat), axis=1)
    e = 1e-10  # numerical stability
    regularization_loss = -0.5 * tf.reduce_sum(
        1 + tf.log(e + tf.square(sigma)) - tf.square(mean) - tf.square(sigma),
        axis=1)

    return tf.reduce_mean(regularization_loss + reconstruction_loss)


vae = Model(input_images, decoded)
vae.compile(optimizer='adam', loss=vae_loss)
vae.summary()

number_of_epochs = int(sys.argv[1] if len(sys.argv) >= 2 else '100')

try:
    vae.fit(
        x_train,
        x_train,
        shuffle=True,
        epochs=number_of_epochs,
        batch_size=128,
        validation_data=(x_test, x_test))
except KeyboardInterrupt:
    pass

# Sample images for display and visualization
original_images = x_test[:10]
reconstructed_images = vae.predict(original_images)

display_images = 10
plot.switch_backend('Agg')
plot.figure(figsize=(20, 4))
for i in range(display_images):
    axis = plot.subplot(2, display_images, i + 1)
    plot.imshow(original_images[i].reshape(original_shape))
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

    axis = plot.subplot(2, display_images, display_images + i + 1)
    plot.imshow(reconstructed_images[i].reshape(original_shape))
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

print('Saving fig.png')
plot.savefig('fig.png')

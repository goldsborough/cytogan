#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10, mnist
from keras.layers import Dense, Input, Lambda
from keras.models import Model


def mnist_data():
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))
    return x_train, x_test, (28, 28, 1)


def cifar_data():
    (x_train, _), (x_test, _) = cifar10.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))
    return x_train, x_test, (32, 32, 3)


x_train, x_test, original_shape = cifar_data()

latent_size = 128
layer_size = 512
input_shape = x_train[0].shape
flat_input_shape = np.prod(input_shape)

input_images = Input([flat_input_shape])

dense_enc = Dense(layer_size, activation='relu')(input_images)
mean = Dense(latent_size)(dense_enc)
log_sigma = Dense(latent_size)(dense_enc)
sigma = tf.exp(log_sigma)


def latent_op(tensors):
    mean, log_sigma = tensors
    noise = tf.truncated_normal(
        shape=[tf.shape(mean)[0], latent_size], stddev=0.1)
    return mean + tf.exp(log_sigma) * noise


latent = Lambda(latent_op)([mean, log_sigma])
dense_dec = Dense(layer_size, activation='relu')(latent)
decoded = Dense(flat_input_shape, activation='sigmoid')(dense_dec)


def vae_loss(y_true, y_pred):
    reconstruction_loss = tf.reduce_sum(tf.square(y_true - y_pred), axis=1)
    e = 1e-10  # numerical stability
    regularization_loss = -0.5 * tf.reduce_sum(
        1 + tf.log(e + tf.square(sigma)) - tf.square(mean) - tf.square(sigma),
        axis=1)

    return tf.reduce_mean(regularization_loss + reconstruction_loss)


vae = Model(input_images, decoded)
vae.compile(optimizer='adam', loss=vae_loss)

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

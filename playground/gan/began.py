import os

import keras.backend as K
import keras.losses
import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import (Conv2D, Dense, Flatten, Input, Lambda, Reshape)
from keras.models import Model
from keras.optimizers import Adam

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(session)

# Supress warnings about wrong compilation of TensorFlow.
tf.logging.set_verbosity(tf.logging.ERROR)

noise_size = 64
latent_size = 64
image_shape = (28, 28, 1)


def upsampling_nn(images):
    new_size = [2 * int(size) for size in images.shape[1:-1]]
    return tf.image.resize_nearest_neighbor(images, size=new_size)


UpSamplingNN = Lambda(upsampling_nn)
reconstruction_error = keras.losses.mean_absolute_error

## G

z = Input(shape=[noise_size])
# 100 x 1 x 1
G = Dense(7 * 7 * 128, activation='elu')(z)
G = Reshape((7, 7, 128))(G)
# 7 x 7 x 128
G = Conv2D(128, (3, 3), padding='same', activation='elu')(G)
G = Conv2D(128, (3, 3), padding='same', activation='elu')(G)
# 7 x 7 x 128
G = UpSamplingNN(G)
G = Conv2D(128, (3, 3), padding='same', activation='elu')(G)
G = Conv2D(128, (3, 3), padding='same', activation='elu')(G)
# 14 x 14 x 128
G = UpSamplingNN(G)
G = Conv2D(128, (3, 3), padding='same', activation='elu')(G)
G = Conv2D(128, (3, 3), padding='same', activation='elu')(G)
# 28 x 28 x 128
G = Conv2D(1, (3, 3), padding='same')(G)

# 28 x 28 x 1

## E


def add_noise(images):
    return images + tf.random_normal(tf.shape(images), mean=0.0, stddev=0.1)


x = Input(shape=image_shape)
E = Lambda(add_noise)(x)
# 28 x 28 x 1
E = Conv2D(128, (3, 3), padding='same', activation='elu')(E)
E = Conv2D(128, (3, 3), padding='same', activation='elu')(E)
E = Conv2D(128, (3, 3), padding='same', activation='elu')(E)
# 28 x 28 x 128
E = Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='elu')(E)
E = Conv2D(256, (3, 3), padding='same', activation='elu')(E)
E = Conv2D(256, (3, 3), padding='same', activation='elu')(E)
# 14 x 14 x 256
E = Conv2D(384, (3, 3), strides=(2, 2), padding='same', activation='elu')(E)
E = Conv2D(384, (3, 3), padding='same', activation='elu')(E)
E = Conv2D(384, (3, 3), padding='same', activation='elu')(E)
E = Flatten()(E)
# 7 x 7 x 384

H = Dense(latent_size)(E)
# 100 x 1 x 1

## D

D = Dense(7 * 7 * 128)(H)
D = Reshape((7, 7, 128))(D)
# 7 x 7 x 128
D = Conv2D(128, (3, 3), padding='same', activation='elu')(D)
D = Conv2D(128, (3, 3), padding='same', activation='elu')(D)
# 7 x 7 x 128
D = UpSamplingNN(D)
D = Conv2D(128, (3, 3), padding='same', activation='elu')(D)
D = Conv2D(128, (3, 3), padding='same', activation='elu')(D)
# 14 x 14 x 128
D = UpSamplingNN(D)
D = Conv2D(128, (3, 3), padding='same', activation='elu')(D)
D = Conv2D(128, (3, 3), padding='same', activation='elu')(D)
# 28 x 28 x 128
D = Conv2D(1, (3, 3), padding='same')(D)
# 28 x 28 x 1

discriminator = Model(x, D)
discriminator.compile(
    loss=reconstruction_error,
    optimizer=Adam(lr=1e-5, beta_1=0.5, decay=1e-10))

generator = Model(z, G)


def generator_loss(_, reconstructions):
    return reconstruction_error(generator.output, reconstructions)


discriminator.trainable = False
gan = Model(z, discriminator(G))
gan.compile(
    loss=generator_loss, optimizer=Adam(lr=1e-5, beta_1=0.5, decay=1e-10))
discriminator.trainable = True

generator.summary()
discriminator.summary()

data = input_data.read_data_sets('MNIST_data').train.images
data = data.reshape(-1, 28, 28, 1) * 2 - 1

number_of_epochs = 200
batch_size = 32
diversity = 0.75
lambda_k = 1e-3
E = 1e-10
k = E


def sample_noise(size):
    return np.random.randn(size, noise_size)


def sample_images(filename, amount=25, rows=5, columns=5):
    images = generator.predict(sample_noise(amount))
    print(images.min(), images.max())
    images = np.clip((images + 1) / 2.0, 0, 1)
    plot.switch_backend('Agg')
    plot.figure(figsize=(10, 4))
    for i in range(amount):
        axis = plot.subplot(rows, columns, i + 1)
        plot.imshow(images[i].reshape(28, 28), cmap='gray')
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    print('Saving {0}'.format(filename))
    plot.savefig(filename)


try:
    for epoch in range(number_of_epochs):
        print('Epoch: {0}/{1}'.format(epoch + 1, number_of_epochs))
        for batch_start in range(0, len(data) - batch_size + 1, batch_size):
            fake_images = generator.predict(sample_noise(batch_size))

            real_images = data[batch_start:batch_start + batch_size]
            real_energy = discriminator.train_on_batch(real_images,
                                                       real_images)

            fake_energy_scaled = discriminator.train_on_batch(
                fake_images,
                fake_images,
                sample_weight=(-k * np.ones(batch_size)))
            d_loss = real_energy + fake_energy_scaled

            noise = sample_noise(batch_size)
            dummy_labels = np.zeros_like(fake_images)
            fake_energy = g_loss = gan.train_on_batch(noise, dummy_labels)

            equilibrium = diversity * real_energy - fake_energy
            k = np.clip(k + lambda_k * equilibrium, E, 1)
            convergence = real_energy + np.abs(equilibrium)

            batch_index = batch_start // batch_size + 1
            message = '\rBatch: {0} | D: {1:.10f} | G: {2:.10f} | ' + \
                      'k: {3:.6f} | M: {4:.8f}'
            print(
                message.format(batch_index, d_loss, g_loss, k, convergence),
                end='')
        print()
        sample_images('began-samples/epoch-{0}.png'.format(epoch + 1))
        np.random.shuffle(data)
except KeyboardInterrupt:
    print()

print('Training complete!')
sample_images('began-samples/final.png')

import os

import keras.backend as K
import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from keras.layers import (BatchNormalization, Concatenate, Conv2D, Dense,
                          Flatten, Input, LeakyReLU, Reshape, UpSampling2D)
from keras.models import Model
from keras.optimizers import Adam
from sklearn.manifold import TSNE
from tensorflow.examples.tutorials.mnist import input_data

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(session)

# Supress warnings about wrong compilation of TensorFlow.
tf.logging.set_verbosity(tf.logging.ERROR)

latent_size = 64

## G

z = Input(shape=[latent_size], name='noise')
# 1 x 1 x 256
G = Dense(7 * 7 * 256)(z)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)
G = Reshape((7, 7, 256))(G)
# 7 x 7 x 256
G = UpSampling2D()(G)
G = Conv2D(128, (5, 5), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)
# 14 x 14 x 128
G = UpSampling2D()(G)
G = Conv2D(64, (4, 4), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)
# 28 x 28 x 64
G = Conv2D(32, (5, 5), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)
# 28 x 28 x 32
G = Conv2D(1, (1, 1), padding='same', activation='tanh')(G)
# 28 x 28 x 1

## E

x = Input(shape=(28, 28, 1))
E = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(x)
E = BatchNormalization(momentum=0.9)(E)
E = LeakyReLU(alpha=0.2)(E)
# 14 x 14 x 32
E = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(E)
E = BatchNormalization(momentum=0.9)(E)
E = LeakyReLU(alpha=0.2)(E)
# 7 x 7 x 64
E = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(E)
E = BatchNormalization(momentum=0.9)(E)
E = LeakyReLU(alpha=0.2)(E)
# 4 x 4 x 128
E = Conv2D(256, (5, 5), padding='same')(E)
E = BatchNormalization(momentum=0.9)(E)
E = LeakyReLU(alpha=0.2)(E)
# 4 x 4 x 256
E = Flatten()(E)
# 4096
E = Dense(latent_size)(E)

## D(x)

x_D = Input(shape=(28, 28, 1), name='x_D')
z_D = Input(shape=[latent_size], name='z_D')
x_flat = Flatten()(x_D)
D = Concatenate()([x_flat, z_D])

D = Dense(1024)(D)
D = LeakyReLU(alpha=0.2)(D)

D = Dense(256)(D)
D = LeakyReLU(alpha=0.2)(D)

D = Dense(128)(D)
D = LeakyReLU(alpha=0.2)(D)

D = Dense(32)(D)
D = LeakyReLU(alpha=0.2)(D)

D = Dense(1, activation='sigmoid')(D)

generator = Model(z, G)
encoder = Model(x, E)
autoencoder = Model(x, generator(E))
discriminator = Model([x_D, z_D], D)

discriminator.compile(
    loss='binary_crossentropy', optimizer=Adam(lr=1e-4, beta_1=0.5))

discriminator.trainable = False
fake_gan = Model(z, discriminator([G, z]))
fake_gan.compile(
    loss='binary_crossentropy', optimizer=Adam(lr=1e-4, beta_1=0.5))

real_gan = Model(x, discriminator([x, E]))
real_gan.compile(
    loss='binary_crossentropy', optimizer=Adam(lr=1e-4, beta_1=0.5))
discriminator.trainable = True

generator.summary()
discriminator.summary()

mnist = input_data.read_data_sets('MNIST_data')
data = mnist.train.images
data = data.reshape(-1, 28, 28, 1) * 2 - 1

number_of_epochs = 20
batch_size = 100


def sample_noise(size):
    return np.random.randn(size, latent_size)


def smooth_labels(size):
    return np.random.uniform(low=0.8, high=1.0, size=size)


try:
    for epoch in range(number_of_epochs):
        print('Epoch: {0}/{1}'.format(epoch + 1, number_of_epochs))
        for batch_start in range(0, len(data) - batch_size + 1, batch_size):
            fake_latent = sample_noise(batch_size)
            fake_images = generator.predict(fake_latent)

            real_images = data[batch_start:batch_start + batch_size]
            real_latent = encoder.predict_on_batch(real_images)

            all_images = np.concatenate([fake_images, real_images], axis=0)
            all_latent = np.concatenate([fake_latent, real_latent], axis=0)

            all_images += np.random.normal(0, 0.1, all_images.shape)

            labels = np.zeros(len(all_images))
            labels[batch_size:] = smooth_labels(batch_size)
            d_loss = discriminator.train_on_batch([all_images, all_latent],
                                                  labels)

            e_loss = real_gan.train_on_batch(real_images, np.zeros(batch_size))

            fake_latent = sample_noise(batch_size)
            g_loss = fake_gan.train_on_batch(fake_latent, np.ones(batch_size))

            batch_index = batch_start // batch_size + 1
            message = '\rBatch: {0} | D: {1:.10f} | G: {2:.10f} | {3:.10f}'
            print(message.format(batch_index, d_loss, g_loss, e_loss), end='')
        print()
        np.random.shuffle(data)
except KeyboardInterrupt:
    print()

print('Training complete!')

if not os.path.exists('bigan'):
    os.makedirs('bigan')

display_images = 100
images = generator.predict(sample_noise(display_images))
images = (images + 1) / 2
plot.switch_backend('Agg')
plot.figure(figsize=(10, 4))
for i in range(display_images):
    axis = plot.subplot(10, 10, i + 1)
    plot.imshow(images[i].reshape(28, 28), cmap='gray')
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
print('Saving bigan/generative.png')
plot.savefig('bigan/generative.png')

display_images = 20
original_images, _ = mnist.test.next_batch(display_images)
original_images = original_images.reshape(-1, 28, 28, 1) * 2 - 1
reconstructed_images = autoencoder.predict_on_batch(original_images)
reconstructed_images = (reconstructed_images + 1) / 2
plot.figure(figsize=(20, 4))
for i in range(display_images):
    axis = plot.subplot(2, display_images, i + 1)
    plot.imshow(original_images[i].reshape(28, 28), cmap='gray')
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

    axis = plot.subplot(2, display_images, display_images + i + 1)
    plot.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

print('Saving bigan/reconstructions.png')
plot.savefig('bigan/reconstructions.png')

display_images = 100
original_images, labels = mnist.test.next_batch(display_images)
original_images = original_images.reshape(-1, 28, 28, 1) * 2 - 1
latent_codes = encoder.predict(original_images)
plot.figure(figsize=(10, 10))
transformed = TSNE(2, perplexity=16, init='pca').fit_transform(latent_codes)
plot.scatter(transformed[:, 0], transformed[:, 1], c=labels, cmap='plasma')
print('Saving bigan/latent-space.png')
plot.savefig('bigan/latent-space.png')

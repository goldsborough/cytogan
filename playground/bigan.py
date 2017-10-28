import keras.backend as K
import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from keras.layers import (BatchNormalization, Concatenate, Conv2D, Dense,
                          Dropout, Flatten, Input, Lambda, LeakyReLU, Reshape,
                          UpSampling2D)
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.examples.tutorials.mnist import input_data

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(session)

# Supress warnings about wrong compilation of TensorFlow.
tf.logging.set_verbosity(tf.logging.ERROR)

latent_size = 64

## G

noise = Input(shape=[latent_size], name='noise')
# 1 x 1 x 256
G = Dense(7 * 7 * 256)(noise)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.1)(G)
G = Reshape((7, 7, 256))(G)
# 7 x 7 x 256
G = UpSampling2D()(G)
G = Conv2D(128, (4, 4), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.1)(G)
# 14 x 14 x 128
G = Conv2D(64, (4, 4), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.1)(G)
# 14 x 14 x 64
G = UpSampling2D()(G)
G = Conv2D(32, (4, 4), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.1)(G)
# 28 x 28 x 32
G = Conv2D(32, (5, 5), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.1)(G)
# 28 x 28 x 32
G = Conv2D(32, (1, 1), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.1)(G)
# 28 x 28 x 32
G = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(G)
# 28 x 28 x 1

## E

R = Input(shape=(28, 28, 1), name='R')
# 28 x 28 x 1
E = Conv2D(32, (5, 5), padding='valid')(R)
E = LeakyReLU(alpha=0.1)(E)
# 24 x 24 x 32
E = Conv2D(64, (4, 4), strides=(2, 2), padding='valid')(E)
E = LeakyReLU(alpha=0.1)(E)
# 11 x 11 x 64
E = Conv2D(128, (4, 4), padding='valid')(E)
E = LeakyReLU(alpha=0.1)(E)
# 8 x 8 x 128
E = Conv2D(256, (4, 4), strides=(2, 2), padding='valid')(E)
E = LeakyReLU(alpha=0.1)(E)
# 3 x 3 x 256
E = Conv2D(512, (3, 3), padding='valid')(E)
E = LeakyReLU(alpha=0.1)(E)
# 1 x 1 x 512
E = Conv2D(512, (1, 1), padding='valid')(E)
E = LeakyReLU(alpha=0.1)(E)
# 1 x 1 x 512
E = Conv2D(128, (1, 1), padding='valid')(E)
E = Flatten()(E)

# 1 x 1 x 128


def reparameterize(logits):
    mean = Dense(latent_size)(logits)
    log_stddev = Dense(latent_size)(logits)
    noise = tf.truncated_normal(
        shape=[tf.shape(mean)[0], latent_size], stddev=0.1)
    return mean + tf.exp(log_stddev) * noise


E = Lambda(reparameterize)(E)

## D(x)

x = Input(shape=(28, 28, 1), name='x')
# 28 x 28 x 1
D_x = Dropout(0.2)(x)
D_x = Conv2D(32, (5, 5), padding='valid')(x)
D_x = LeakyReLU(alpha=0.1)(D_x)
# 24 x 24 x 32
D_x = Dropout(0.5)(D_x)
D_x = Conv2D(64, (4, 4), strides=(2, 2), padding='valid')(D_x)
D_x = LeakyReLU(alpha=0.1)(D_x)
# 11 x 11 x 64
D_x = Dropout(0.5)(D_x)
D_x = Conv2D(128, (4, 4), padding='valid')(D_x)
D_x = LeakyReLU(alpha=0.1)(D_x)
# 8 x 8 x 128
D_x = Dropout(0.5)(D_x)
D_x = Conv2D(256, (4, 4), strides=(2, 2), padding='valid')(D_x)
D_x = LeakyReLU(alpha=0.1)(D_x)
# 3 x 3 x 256
D_x = Dropout(0.5)(D_x)
D_x = Conv2D(512, (3, 3), padding='valid')(D_x)
D_x = LeakyReLU(alpha=0.1)(D_x)
# 1 x 1 x 512
D_x = Flatten()(D_x)
# 512

## D(z)

z = Input(shape=[latent_size], name='z')
# 1 x 1 x 64
D_z = Dropout(0.2)(z)
D_z = Dense(512, activation='relu')(z)
# 1 x 1 x 512
D_z = Dropout(0.5)(D_z)
D_z = Dense(512, activation='relu')(D_z)
# 512

## D(x, z)

D = Concatenate()([D_x, D_z])
# 1 x 1 x 1024
D = Dropout(0.5)(D)
D = Dense(1024, activation='relu')(D)
# 1 x 1 x 1024
D = Dropout(0.5)(D)
D = Dense(1024, activation='relu')(D)
# 1 x 1 x 1024
D = Dense(1, activation='sigmoid')(D)

generator = Model(noise, G)
encoder = Model(R, E)
autoencoder = Model(R, generator(E))

discriminator = Model([x, z], D)
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=1e-4, beta_1=0.5, beta_2=1e-3, decay=2e-7))

# X = Concatenate(axis=0, name='X')([G, R])
# Z = Concatenate(axis=0, name='Z')([noise, E])

# discriminator.trainable = False
# gan = Model([noise, R], discriminator([X, Z]))
# gan.compile(
#     loss='binary_crossentropy',
#     optimizer=Adam(lr=1e-4, beta_1=0.5, beta_2=1e-3, decay=1e-7))
# discriminator.trainable = True

discriminator.trainable = False
fake_gan = Model(noise, discriminator([G, noise]))
fake_gan.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=1e-4, beta_1=0.5, beta_2=1e-3, decay=1e-7))

real_gan = Model(R, discriminator([R, E]))
real_gan.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=1e-4, beta_1=0.5, beta_2=1e-3, decay=1e-7))
discriminator.trainable = True

generator.summary()
discriminator.summary()

mnist = input_data.read_data_sets('MNIST_data')
data = mnist.train.images
data = data.reshape(-1, 28, 28, 1) * 2 - 1

number_of_epochs = 100
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
            all_images += np.random.normal(0, 0.1, all_images.shape)

            all_latent = np.concatenate([fake_latent, real_latent], axis=0)

            labels = np.zeros(len(all_images))
            labels[batch_size:] = smooth_labels(batch_size)
            d_loss = discriminator.train_on_batch([all_images, all_latent],
                                                  labels)

            e_loss = real_gan.train_on_batch(real_images, np.zeros(batch_size))

            fake_latent = sample_noise(batch_size)
            g_loss = fake_gan.train_on_batch(fake_latent, np.ones(batch_size))

            batch_index = batch_start // batch_size + 1
            message = '\rBatch: {0} | D: {1:.10f} | G: {2:.10f}'
            print(message.format(batch_index, d_loss, g_loss), end='')
        print()
        np.random.shuffle(data)
except KeyboardInterrupt:
    print()

print('Training complete!')

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
print('Saving generative.png')
plot.savefig('generative.png')

display_images = 20
original_images, _ = mnist.test.next_batch(display_images)
original_images = original_images.reshape(-1, 28, 28, 1)
reconstructed_images = autoencoder.predict_on_batch(original_images)
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

print('Saving reconstructions.png')
plot.savefig('reconstructions.png')

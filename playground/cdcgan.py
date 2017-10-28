import keras.backend as K
import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Flatten, Input, LeakyReLU, Reshape, UpSampling2D,
                          Concatenate)
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.examples.tutorials.mnist import input_data

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(session)

# Supress warnings about wrong compilation of TensorFlow.
tf.logging.set_verbosity(tf.logging.ERROR)

noise_size = 100
conditional_size = 10
image_shape = (28, 28, 1)

## G

z = Input(shape=[noise_size])
c_G = Input(shape=[conditional_size])
G = Concatenate(axis=1)([z, c_G])

G = Dense(7 * 7 * 256)(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)
G = Reshape((7, 7, 256))(G)

G = UpSampling2D()(G)
G = Conv2D(128, (5, 5), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)

G = UpSampling2D()(G)
G = Conv2D(64, (5, 5), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)

G = Conv2D(32, (5, 5), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)

G = Conv2D(1, (5, 5), padding='same')(G)
G = Activation('tanh')(G)
assert G.shape[1:] == image_shape, G

## D

x = Input(shape=image_shape)
c_D = Input(shape=[conditional_size])
x_flat = Reshape([np.prod(image_shape)])(x)
D = Concatenate(axis=1)([x_flat, c_D])
D = Dense(28 * 28 * 2, activation='relu')(D)
D = Reshape((28, 28, 2))(D)

D = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(D)
D = LeakyReLU(alpha=0.2)(D)

D = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(D)
D = LeakyReLU(alpha=0.2)(D)

D = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(D)
D = LeakyReLU(alpha=0.2)(D)

D = Conv2D(256, (5, 5), padding='same')(D)
D = LeakyReLU(alpha=0.2)(D)

D = Flatten()(D)
D = Dense(1)(D)
D = Activation('sigmoid')(D)

generator = Model([z, c_G], G)

discriminator = Model([x, c_D], D)
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=5e-4, beta_1=0.5, decay=2e-7))

discriminator.trainable = False
gan = Model([z, c_G], discriminator([G, c_G]))
gan.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=2e-4, beta_1=0.5, decay=1e-7))
discriminator.trainable = True

generator.summary()
discriminator.summary()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True).train
train_images = mnist.images.reshape(-1, 28, 28, 1) * 2 - 1
train_labels = mnist.labels

number_of_epochs = 30
batch_size = 256
label_smoothing = 0.9


def sample_noise(size):
    return np.random.randn(size, noise_size)


def smooth_labels(size):
    return np.random.uniform(low=0.8, high=1.0, size=size)


def sample_conditional(size):
    return np.random.multinomial(1, [0.1] * 10, size=size)


try:
    number_of_batches = len(train_images) - batch_size + 1
    for epoch in range(number_of_epochs):
        print('Epoch: {0}/{1}'.format(epoch + 1, number_of_epochs))
        for batch_start in range(0, number_of_batches, batch_size):

            real_images = train_images[batch_start:batch_start + batch_size]
            conditionals = train_labels[batch_start:batch_start + batch_size]

            generated_images = generator.predict(
                [sample_noise(batch_size), conditionals])

            all_images = np.concatenate(
                [generated_images, real_images], axis=0)
            all_images += np.random.normal(0, 0.1, all_images.shape)
            all_conditionals = np.concatenate(
                [conditionals, conditionals], axis=0)

            labels = np.zeros(len(all_images))
            labels[batch_size:] = smooth_labels(batch_size)
            d_loss = discriminator.train_on_batch(
                [all_images, all_conditionals], labels)

            conditionals = sample_conditional(batch_size)
            labels = np.ones(batch_size)
            g_loss = gan.train_on_batch(
                [sample_noise(batch_size), conditionals], labels)

            batch_index = batch_start // batch_size + 1
            message = '\rBatch: {0} | D: {1:.10f} | G: {2:.10f}'
            print(message.format(batch_index, d_loss, g_loss), end='')
        print()
        np.random.shuffle(train_images)
except KeyboardInterrupt:
    print()

print('Training complete!')

display_images = 100
noise = sample_noise(display_images)
conditionals = sample_conditional(display_images)
images = generator.predict([noise, conditionals])
images = (images + 1) / 2
plot.switch_backend('Agg')
plot.figure(figsize=(10, 4))
for i in range(display_images):
    axis = plot.subplot(10, 10, i + 1)
    plot.imshow(images[i].reshape(28, 28), cmap='gray')
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
print('Saving fig.png')
plot.savefig('fig.png')

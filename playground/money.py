import keras.backend as K
import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Flatten, Input, LeakyReLU, Reshape, UpSampling2D)
from keras.models import Model
from keras.optimizers import Adam
import scipy.misc

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(session)

# Supress warnings about wrong compilation of TensorFlow.
tf.logging.set_verbosity(tf.logging.ERROR)

noise_size = 100

## G

z = Input(shape=[noise_size])

G = Dense(8 * 4 * 256)(z)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)
G = Reshape((4, 8, 256))(G)

G = UpSampling2D()(G)
G = Conv2D(128, (5, 5), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)

G = UpSampling2D()(G)
G = Conv2D(64, (5, 5), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)

G = UpSampling2D()(G)
G = Conv2D(32, (5, 5), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)

G = UpSampling2D()(G)
G = Conv2D(32, (5, 5), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)

G = Conv2D(3, (5, 5), padding='same')(G)
G = Activation('tanh')(G)

## D

x = Input(shape=(64, 128, 3))
D = Conv2D(16, (5, 5), strides=(2, 2), padding='same')(x)
D = LeakyReLU(alpha=0.2)(D)

D = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(D)
D = LeakyReLU(alpha=0.2)(D)

D = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(D)
D = LeakyReLU(alpha=0.2)(D)

D = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(D)
D = LeakyReLU(alpha=0.2)(D)

D = Flatten()(D)
D = Dense(1)(D)
D = Activation('sigmoid')(D)

generator = Model(z, G)

discriminator = Model(x, D)
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=5e-4, beta_1=0.5, decay=2e-7))

discriminator.trainable = False
gan = Model(z, discriminator(G))
gan.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=2e-4, beta_1=0.5, decay=1e-7))
discriminator.trainable = True

generator.summary()
discriminator.summary()

image = scipy.misc.imread('money.jpg')
image = (image / 127.5) - 1
data = [image.reshape(64, 128, 3)] * 10000

number_of_epochs = 10
batch_size = 128


def noise(size):
    return np.random.randn(size, noise_size)


def smooth_labels(size):
    return np.random.uniform(low=0.8, high=1.0, size=size)


try:
    for epoch in range(number_of_epochs):
        print('Epoch: {0}/{1}'.format(epoch + 1, number_of_epochs))
        for batch_start in range(0, len(data) - batch_size + 1, batch_size):
            generated_images = generator.predict(noise(batch_size))
            real_images = data[batch_start:batch_start + batch_size]
            all_images = np.concatenate(
                [generated_images, real_images], axis=0)
            all_images += np.random.normal(0, 0.1, all_images.shape)

            labels = np.zeros(len(all_images))
            labels[batch_size:] = smooth_labels(batch_size)
            d_loss = discriminator.train_on_batch(all_images, labels)

            labels = np.ones(batch_size)
            g_loss = gan.train_on_batch(noise(batch_size), labels)

            batch_index = batch_start // batch_size + 1
            message = '\rBatch: {0} | D: {1:.10f} | G: {2:.10f}'
            print(message.format(batch_index, d_loss, g_loss), end='')
        print()
        np.random.shuffle(data)
except KeyboardInterrupt:
    print()

print('Training complete!')

display_images = 16
images = generator.predict(noise(display_images))
images = (images + 1) / 2
plot.switch_backend('Agg')
plot.figure(figsize=(10, 4))
for i in range(display_images):
    axis = plot.subplot(4, 4, i + 1)
    plot.imshow(images[i])
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
print('Saving fig.png')
plot.savefig('fig.png')

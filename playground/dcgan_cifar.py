import keras.backend as K
import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Flatten, Input, LeakyReLU, Reshape, UpSampling2D)
from keras.models import Model
from keras.optimizers import Adam

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(session)

# Supress warnings about wrong compilation of TensorFlow.
tf.logging.set_verbosity(tf.logging.ERROR)

noise_size = 100

## G

z = Input(shape=[noise_size])

G = Dense(7 * 7 * 256, input_dim=100)(z)
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

## D

x = Input(shape=(28, 28, 1))
D = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(x)
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
# 28 x 28 x 1

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

(x_train, _), (x_test, _) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1) / 127.5 - 1
x_test = np.expand_dims(x_train, axis=-1) / 127.5 - 1

number_of_epochs = 100
batch_size = 256
label_smoothing = 0.9


def noise(size):
    return np.random.randn(size, noise_size)


try:
    for epoch in range(number_of_epochs):
        print('Epoch: {0}/{1}'.format(epoch + 1, number_of_epochs))
        for batch_start in range(0, len(x_train), batch_size):
            generated_images = generator.predict(noise(batch_size))
            real_images = x_train[batch_start:batch_start + batch_size]
            all_images = np.concatenate(
                [generated_images, real_images], axis=0)
            noisy_images = all_images + np.random.normal(
                0, 0.1, all_images.shape)

            labels = np.zeros(len(all_images))
            labels[batch_size:] = label_smoothing
            d_loss = discriminator.train_on_batch(noisy_images, labels)

            labels = np.ones(batch_size) * label_smoothing
            g_loss = gan.train_on_batch(noise(batch_size), labels)

            batch_index = batch_start // batch_size + 1
            message = '\rBatch: {0} | D: {1:.10f} | G: {2:.10f}'
            print(message.format(batch_index, d_loss, g_loss), end='')
        print()
        np.random.shuffle(x_train)
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
    plot.imshow(images[i].reshape(28, 28), cmap='gray')
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
print('Saving fig.png')
plot.savefig('fig.png')

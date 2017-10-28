import keras.backend as K
import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Flatten, Input, LeakyReLU, Reshape, UpSampling2D)
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.examples.tutorials.mnist import input_data

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

generator = Model(z, G)
discriminator = Model(x, D)


def d_loss(_, logits):
    generated_logits, real_logits = tf.split(logits, 2)
    loss = K.mean(generated_logits) - K.mean(real_logits)

    generated_images, real_images = tf.split(discriminator.input, 2)
    epsilon = tf.random_uniform(shape=K.shape(generated_images))
    mix = epsilon * real_images + (1 - epsilon) * generated_images
    gradients = K.gradients(discriminator(mix), mix)[0]
    slopes = K.sqrt(K.sum(K.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = 10 * K.mean(K.square(slopes - 1), axis=0)

    return loss + gradient_penalty


def g_loss(_, logits):
    return -K.mean(logits)


discriminator.compile(
    loss=d_loss, optimizer=Adam(lr=1e-4, beta_1=0.5, beta_2=0.9))

discriminator.trainable = False
gan = Model(z, discriminator(G))
gan.compile(loss=g_loss, optimizer=Adam(lr=1e-4, beta_1=0.5, beta_2=0.9))
discriminator.trainable = True

generator.summary()
discriminator.summary()

data = input_data.read_data_sets('MNIST_data').train.images
data = data.reshape(-1, 28, 28, 1) * 2 - 1

number_of_epochs = 100
batch_size = 256
n_critic = 5


def noise(size):
    return np.random.randn(size, noise_size)


try:
    for epoch in range(number_of_epochs):
        print('Epoch: {0}/{1}'.format(epoch + 1, number_of_epochs))
        for batch_start in range(0, len(data) - batch_size + 1, batch_size):
            for _ in range(n_critic):
                generated_images = generator.predict(noise(batch_size))
                real_images = data[batch_start:batch_start + batch_size]
                assert len(generated_images) == len(real_images)
                all_images = np.concatenate(
                    [generated_images, real_images], axis=0)
                noisy_images = all_images + np.random.normal(
                    0, 0.1, all_images.shape)

                labels = np.zeros(len(all_images))
                labels[batch_size:] = 1
                d_loss = discriminator.train_on_batch(noisy_images, labels)

            labels = np.ones(batch_size)
            g_loss = gan.train_on_batch(noise(batch_size), labels)

            batch_index = batch_start // batch_size + 1
            message = '\rBatch: {0} | D: {1:.10f} | G: {2:.10f}'
            print(message.format(batch_index, -d_loss, -g_loss), end='')
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
    plot.imshow(images[i].reshape(28, 28), cmap='gray')
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
print('Saving fig.png')
plot.savefig('fig.png')

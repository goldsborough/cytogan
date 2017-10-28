import keras.backend as K
import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                          Dense, Flatten, Input, LeakyReLU, Reshape,
                          UpSampling2D)
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.examples.tutorials.mnist import input_data

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(session)

# Supress warnings about wrong compilation of TensorFlow.
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(42)

noise_size = 100
latent_size = 10

## G

z = Input(shape=[noise_size])
c = Input(shape=[latent_size])
G = Concatenate()([z, c])

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

Q = Dense(latent_size, activation='softmax')(D)
D = Dense(1, activation='sigmoid')(D)


def mutual_information(prior_c, c_given_x):
    h_c = K.categorical_crossentropy(prior_c, prior_c)
    h_c_given_x = K.categorical_crossentropy(prior_c, c_given_x)
    return K.mean(h_c_given_x - h_c)


generator = Model([z, c], G, name='G')

discriminator = Model(x, D, name='D')
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=5e-4, beta_1=0.5, decay=2e-7))

# x = G(z, c)
q = Model(x, Q, name='Q')
q.compile(
    loss=mutual_information, optimizer=Adam(lr=2e-4, beta_1=0.5, decay=2e-7))

discriminator.trainable = False
q.trainable = False
infogan = Model([z, c], [discriminator(G), q(G)], name='InfoGAN')
infogan.compile(
    loss=['binary_crossentropy', mutual_information],
    optimizer=Adam(lr=2e-4, beta_1=0.5, decay=1e-7))

generator.summary()
discriminator.summary()

data = input_data.read_data_sets('MNIST_data').train.images
data = data.reshape(-1, 28, 28, 1) * 2 - 1

number_of_epochs = 20
batch_size = 256


def sample_noise(size):
    return np.random.randn(size, noise_size)


def sample_prior(size):
    return np.random.multinomial(1, [0.1] * 10, size=size)

def smooth_labels(size):
    return np.random.uniform(low=0.8, high=1.0, size=size)

try:
    for epoch in range(number_of_epochs):
        print('Epoch: {0}/{1}'.format(epoch + 1, number_of_epochs))
        for batch_start in range(0, len(data) - batch_size + 1, batch_size):
            noise = sample_noise(batch_size)
            latent_code = sample_prior(batch_size)
            generated_images = generator.predict([noise, latent_code])

            real_images = data[batch_start:batch_start + batch_size]
            assert len(generated_images) == len(real_images)
            all_images = np.concatenate(
                [generated_images, real_images], axis=0)
            all_images += np.random.normal(0, 0.1, all_images.shape)

            labels = np.zeros(len(all_images))
            labels[batch_size:] = smooth_labels(batch_size)
            d_loss = discriminator.train_on_batch(all_images, labels)

            q_loss = q.train_on_batch(generated_images, latent_code)

            labels = np.ones(batch_size)
            noise = sample_noise(batch_size)
            latent_code = sample_prior(batch_size)
            g_loss, _, _ = infogan.train_on_batch([noise, latent_code],
                                                  [labels, latent_code])

            batch_index = batch_start // batch_size + 1
            message = '\rBatch: {0} | D: {1:.10f} | G: {2:.10f} | Q: {3:.10f}'
            print(message.format(batch_index, d_loss, g_loss, q_loss), end='')
        print()
        np.random.shuffle(data)
except KeyboardInterrupt:
    print()

print('Training complete!')

images_per_digit = 5
total_images = 10 * images_per_digit
noise = sample_noise(total_images)
latent_code = np.eye(latent_size).repeat(images_per_digit, axis=0)
images = generator.predict_on_batch([noise, latent_code])
images = (images + 1) / 2
plot.switch_backend('Agg')
plot.figure(figsize=(10, 4))
for i in range(total_images):
    axis = plot.subplot(10, images_per_digit, i + 1)
    plot.imshow(images[i].reshape(28, 28), cmap='gray')
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
print('Saving fig.png')
plot.savefig('fig.png')

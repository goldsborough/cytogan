#!/usr/bin/env python3

import matplotlib.pyplot as plot
import numpy as np
import time

from keras import regularizers
from keras.datasets import mnist
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

encoding_dim = 32
input_vector = Input(shape=(784, ))

# regular

encoded = Dense(encoding_dim, activation='relu')(input_vector)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder_regular = Model(input_vector, decoded)

# sparse

encoded_sparse = Dense(
    encoding_dim,
    activation='relu',
    activity_regularizer=regularizers.l1(10e-3))(input_vector)
decoded_sparse = Dense(784, activation='sigmoid')(encoded_sparse)
autoencoder_sparse = Model(input_vector, decoded_sparse)

# deep

encoded_deep = Dense(128, activation='relu')(input_vector)
encoded_deep = Dense(64, activation='relu')(encoded_deep)
encoded_deep = Dense(32, activation='relu')(encoded_deep)

decoded_deep = Dense(64, activation='relu')(encoded_deep)
decoded_deep = Dense(128, activation='relu')(decoded_deep)
decoded_deep = Dense(784, activation='sigmoid')(decoded_deep)
autoencoder_deep = Model(input_vector, decoded_deep)

# convolutional

input_img = Input(shape=(28, 28, 1))

conv = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
conv = MaxPooling2D((2, 2), padding='same')(conv)
conv = Conv2D(8, (3, 3), activation='relu', padding='same')(conv)
conv = MaxPooling2D((2, 2), padding='same')(conv)
conv = Conv2D(8, (3, 3), activation='relu', padding='same')(conv)
encoded_conv = MaxPooling2D((2, 2), padding='same')(conv)

deconv = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_conv)
deconv = UpSampling2D((2, 2))(deconv)
deconv = Conv2D(8, (3, 3), activation='relu', padding='same')(deconv)
deconv = UpSampling2D((2, 2))(deconv)
deconv = Conv2D(16, (3, 3), activation='relu')(deconv)
deconv = UpSampling2D((2, 2))(deconv)
decoded_conv = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(deconv)

encoder_conv = Model(input_img, encoded_conv)
autoencoder_conv = Model(input_img, decoded_conv)

# Training

autoencoder = autoencoder_conv
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
# x_train = x_train.reshape((len(x_train), -1))
# x_test = x_test.reshape((len(x_test), -1))

# noise_factor = 0.5
# x_train_noisy = x_train + noise_factor * np.random.normal(
#     loc=0.0, scale=1.0, size=x_train.shape)
# x_test_noisy = x_test + noise_factor * np.random.normal(
#     loc=0.0, scale=1.0, size=x_test.shape)
#
# x_train_noisy = np.clip(x_train_noisy, 0., 1.)
# x_test_noisy = np.clip(x_test_noisy, 0., 1.)

start_time = time.time()

try:
    autoencoder.fit(
        x_train,
        x_train,
        epochs=5,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test, x_test))
except KeyboardInterrupt:
    pass

# Inference

print('Training complete! Took: {:.3f}s'.format(time.time() - start_time))

# encoded_imgs = encoder_conv.predict(x_test)
reconstructed_imgs = autoencoder.predict(x_test)

print('Inference complete!')

n = 10
plot.figure(figsize=(20, 4))
for i in range(n):
    axis = plot.subplot(2, n, i + 1)
    plot.imshow(x_test[i].reshape(28, 28))
    plot.gray()
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

    axis = plot.subplot(2, n, i + 1 + n)
    plot.imshow(reconstructed_imgs[i].reshape(28, 28))
    plot.gray()
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

# plot.figure(figsize=(20, 8))
# for i in range(n):
#     axis = plot.subplot(1, n, i + 1)
#     plot.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
#     plot.gray()
#     axis.get_xaxis().set_visible(False)
#     axis.get_yaxis().set_visible(False)

plot.show()

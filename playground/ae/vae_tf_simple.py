#!/usr/bin/env python3

import sys
import time

import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.examples.tutorials import mnist

latent_size = 128
layer_size = 512

input_images = tf.placeholder(dtype=tf.float32, shape=[None, 28 * 28])
batch_size = tf.shape(input_images)[0]

# [n, 784] -> [n, layer_size]
w_enc = tf.get_variable('w_enc', shape=[784, layer_size])
b_enc = tf.get_variable(
    'b_enc', shape=[layer_size], initializer=tf.constant_initializer(0.0))
h = tf.nn.relu(tf.matmul(input_images, w_enc) + b_enc)


# m: [n, layer_size] -> [n, latent_size]
w_m = tf.get_variable('w_m', shape=[layer_size, latent_size])
b_m = tf.get_variable(
    'b_m', shape=[latent_size], initializer=tf.constant_initializer(0.0))
mean = tf.matmul(h, w_m) + b_m

# s: [n, layer_size] -> [n, latent_size]
w_s = tf.get_variable('w_s', shape=[layer_size, latent_size])
b_s = tf.get_variable(
    'b_s', shape=[latent_size], initializer=tf.constant_initializer(0.0))
log_sigma = tf.matmul(h, w_s) + b_s
sigma = tf.exp(log_sigma)

noise = tf.truncated_normal(shape=[batch_size, latent_size], stddev=0.1)
z = mean + sigma * noise


# [n, latent_size] -> [n, layer_size]
w_dec = tf.get_variable('w_dec', shape=[latent_size, layer_size])
b_dec = tf.get_variable(
    'b_dec', shape=[layer_size], initializer=tf.constant_initializer(0.0))
h_intermediate = tf.nn.relu(tf.matmul(z, w_dec) + b_dec)

# [n, layer_size] -> [n, 784]
w_final = tf.get_variable('w_final', shape=[layer_size, 784])
b_final = tf.get_variable(
    'b_final', shape=[784], initializer=tf.constant_initializer(0.0))
decoded = tf.nn.sigmoid(tf.matmul(h_intermediate, w_final) + b_final)

# reconstruction_loss = 0.5 * tf.reduce_mean(
#     tf.square(decoded - input_images), axis=1)

e = 1e-10  # numerical stability
binary_cross_entropies = input_images * tf.log(e + decoded) + \
                        (1 - input_images) * tf.log(e + 1 - decoded)
reconstruction_loss = -tf.reduce_sum(binary_cross_entropies, axis=1)

# regularization_loss = -0.5 * tf.reduce_sum(
#     1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)

regularization_loss = -0.5 * tf.reduce_sum(
    1 + tf.log(e + tf.square(sigma)) - tf.square(mean) - tf.square(sigma),
    axis=1)

loss_raw = tf.reduce_mean(regularization_loss + reconstruction_loss)
loss = tf.clip_by_value(loss_raw, 0, 1000)

data = mnist.input_data.read_data_sets('MNIST_data', one_hot=False)
number_of_epochs = int(sys.argv[1] if len(sys.argv) >= 2 else '100')
batch_size = 128
checkpoint = 1
number_of_batches = data.train.num_examples // batch_size

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
    1e-3,
    global_step=global_step,
    decay_steps=number_of_batches,
    decay_rate=0.995,
    staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimize = optimizer.minimize(loss, global_step=global_step)

with tf.Session() as session:
    tf.global_variables_initializer().run()
    start = time.time()
    try:
        for epoch in range(1, number_of_epochs + 1):
            for _ in range(number_of_batches):
                batch, _ = data.train.next_batch(batch_size)
                _, current_loss, current_lr = session.run(
                    [optimize, loss, learning_rate],
                    feed_dict={input_images: batch})
            if np.isnan(current_loss):
                raise RuntimeError('Loss was NaN')
            if epoch % checkpoint == 0:
                message = 'Epoch {0}/{1} | Loss: {2:.6f} | LR: {3}'
                print(
                    message.format(epoch, number_of_epochs, current_loss,
                                   current_lr))
    except KeyboardInterrupt:
        pass

    print('Training complete! Took {0:.2f}s'.format(time.time() - start))

    # Sample images for display and visualization
    original_images, labels = data.test.next_batch(10000)
    values = session.run(
        [decoded, z], feed_dict={input_images: original_images})
    reconstructed_images, latent_vectors = values

display_images = 10
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

# if latent_size > 2:
#     print('Reducing dimensionality ...')
#     tsne = TSNE(n_components=2)
#     latent_vectors = tsne.fit_transform(latent_vectors)
#     # pca = PCA(n_components=2)
#     # latent_vectors = pca.fit_transform(latent_vectors)
#
# plot.figure(figsize=(10, 10))
# plot.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=labels)
# plot.colorbar()

plot.show()

#!/usr/bin/env python3

import sys
import time

import matplotlib.pyplot as plot
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.examples.tutorials import mnist

flattened_images = tf.placeholder(dtype=tf.float32, shape=[None, 28 * 28])

# [n, 784] -> [n, 32]
w_enc = tf.get_variable('w_enc', shape=[784, 32])
b_enc = tf.get_variable('b_enc', shape=[32])
encoded = tf.nn.relu(tf.matmul(flattened_images, w_enc) + b_enc)

w_dec = tf.get_variable('w_dec', shape=[32, 784])
b_dec = tf.get_variable('b_dec', shape=[784])
decoded = tf.nn.sigmoid(tf.matmul(encoded, w_dec) + b_dec)

loss = tf.reduce_mean(tf.square(decoded - flattened_images))

optimize = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

data = mnist.input_data.read_data_sets('MNIST_data', one_hot=False)
number_of_epochs = int(sys.argv[1] if len(sys.argv) >= 2 else '100')
batch_size = 128
checkpoint = 1
number_of_batches = data.train.num_examples // batch_size


with tf.Session() as session:
    tf.global_variables_initializer().run()
    start = time.time()
    try:
        for epoch in range(1, number_of_epochs + 1):
            for _ in range(number_of_batches):
                batch, _ = data.train.next_batch(batch_size)
                _, loss_value = session.run([optimize, loss], feed_dict={
                    flattened_images: batch
                })
            if epoch % checkpoint == 0:
                message = 'Epoch {0}/{1} | Loss: {2:.6f}'
                print(message.format(epoch, number_of_epochs, loss_value))
    except KeyboardInterrupt:
        pass

    print('Training complete! Took {0:.2f}s'.format(time.time() - start))

    # Sample images for display and visualization
    original_images, labels = data.test.next_batch(1000)
    values = session.run([decoded, encoded], feed_dict={
        flattened_images: original_images
    })
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

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(latent_vectors)

plot.figure(figsize=(10, 10))
plot.title('PCA')
plot.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels)
plot.colorbar()

tsne = TSNE(n_components=2)
reduced_vectors = tsne.fit_transform(latent_vectors)

plot.figure(figsize=(10, 10))
plot.title('t-SNE')
plot.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels)
plot.colorbar()

plot.show()

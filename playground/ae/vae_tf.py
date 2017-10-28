#!/usr/bin/env python3

# conv->max->conv->max -> latent -> conv->up->conv->up->conv

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials import mnist

flattened_images = tf.placeholder(dtype=tf.float32, shape=[None, 28 * 28])
images = tf.reshape(flattened_images, shape=[None, 28, 28, 1])

def conv(out, stride):
    return Conv2D(out, stride)

# [n, 28, 28, 1] -> [n, 28, 28, 16]
w_conv1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 16]))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]))
z_conv1 = tf.nn.conv2d(images, w_conv1, strides=np.ones(4), padding='SAME')
h_conv1 = tf.nn.relu(z_conv1 + b_conv1)

# [n, 28, 28, 16] -> [n, 14, 14, 16]
h_pool1 = tf.nn.max_pool(h_conv1,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME')

# [n, 14, 14, 16] -> [n, 14, 14, 8]
w_conv2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 8]))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[8]))
z_conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=np.ones(4), padding='SAME')
h_conv2 = tf.nn.relu(z_conv2 + b_conv2)

# [n, 14, 14, 8] -> [n, 7, 7, 8]
h_pool2 = tf.nn.max_pool(h_conv2,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME')

# [n, 7, 7, 8] -> [n, 7, 7, 8]
w_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 8, 8]))
b_conv3 = tf.Variable(tf.constant(0.1, shape=[8]))
z_conv3 = tf.nn.conv2d(h_pool2, w_conv3, strides=np.ones(4), padding='SAME')
h_conv3 = tf.nn.relu(z_conv3 + b_conv3)

# [n, 7, 7, 8] -> [n, 4, 4, 8]
h_pool3 = tf.nn.max_pool(h_conv3,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME')

# VAE stuff

# [n, 4, 4, 8] -> [n, 4, 4, 8]
w_deconv1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 8, 8]))
b_deconv1 = tf.Variable(tf.constant(0.1, shape=[8]))
z_deconv1 = tf.nn.conv2d(h_pool3, w_deconv1, strides=np.ones(4), padding='SAME')
h_deconv1 = tf.nn.relu(z_deconv1 + b_deconv1)

# [n, 4, 4, 8] -> [n, 8, 8, 8]
h_up1  = tf.image.resize_images(h_pool3, size=[h_pool3.shape[0], 8, 8, 8])

# [n, 8, 8, 8] -> [n, 8, 8, 8]
w_deconv2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 8, 8]))
b_deconv2 = tf.Variable(tf.constant(0.1, shape=[8]))
z_deconv2 = tf.nn.conv2d(h_up1, w_deconv2, strides=np.ones(4), padding='SAME')
h_deconv2 = tf.nn.relu(z_deconv2 + b_deconv2)

# [n, 8, 8, 8] -> [n, 16, 16, 8]
h_up2  = tf.image.resize_images(h_deconv2, size=[h_deconv2.shape[0], 16, 16, 8])

# [n, 16, 16, 8] -> [n, 14, 14, 16]
w_deconv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 8, 16]))
b_deconv3 = tf.Variable(tf.constant(0.1, shape=[16]))
z_deconv3 = tf.nn.conv2d(h_up2, w_deconv3, strides=np.ones(4), padding='VALID')
h_deconv3 = tf.nn.relu(z_deconv3 + b_deconv3)

# [n, 14, 14, 16] -> [n, 28, 28, 16]
h_up3  = tf.image.resize_images(h_deconv3,
                                size=[h_deconv3.shape[0], 28, 28, 16])

# [n, 28, 28, 16] -> [n, 28, 28, 1]
w_deconv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 1]))
b_deconv4 = tf.Variable(tf.constant(0.1, shape=[1]))
z_deconv4 = tf.nn.conv2d(h_up3, w_deconv4, strides=np.ones(4), padding='SAME')
h_deconv4 = tf.nn.sigmoid(z_deconv4 + b_deconv4)

data = mnist.input_data.read_data_sets('MNIST_data', one_hot=True)
# training_images, test_images = data.train.images, data.test.images
# print(training_images.shape, test_images.shape)

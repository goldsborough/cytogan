#!/usr/bin/env python3

import argparse

from tensorflow.examples.tutorials import mnist

import tensorflow as tf

from cytogan.train import trainer, visualize
from cytogan.models import vae

parser = argparse.ArgumentParser(description='cytogan-mnist')
parser.add_argument('-e', '--epochs', type=int, default=10)
parser.add_argument('-b', '--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr-decay', type=float, default=0.995)
parser.add_argument('--sample', type=int, default=10)
options = parser.parse_args()

print(options)

data = mnist.input_data.read_data_sets('MNIST_data', one_hot=False)
get_batch = lambda n: data.train.next_batch(n)[0].reshape([-1, 28, 28, 1])
number_of_batches = data.train.num_examples // options.batch_size

model = vae.VAE(image_shape=[28, 28, 1])
model.prepare(
    options.lr,
    decay_learning_rate_after=number_of_batches,
    learning_rate_decay=options.lr_decay)

trainer = trainer.Trainer(options.epochs, number_of_batches,
                          options.batch_size)
print(trainer)

with tf.Session() as session:
    trainer.train(model, session, get_batch)
    original_images, labels = data.test.next_batch(options.sample)
    original_images = original_images.reshape(-1, 28, 28, 1)
    latent_vectors, reconstructed_images = model.generate(
        original_images, session)

visualize.reconstructions(original_images, reconstructed_images, gray=True)
visualize.latent_space(latent_vectors, labels)
visualize.show()

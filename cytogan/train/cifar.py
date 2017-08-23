#!/usr/bin/env python3

import argparse

from keras.datasets import cifar10

from cytogan.models import ae, conv_ae, vae
from cytogan.train import trainer, visualize
from cytogan.train.utility import BatchGenerator, Dataset
import numpy as np

parser = argparse.ArgumentParser(description='cytogan-cifar')
parser.add_argument('-e', '--epochs', type=int, default=5)
parser.add_argument('-b', '--batch-size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr-decay', type=float, default=1)
parser.add_argument('-r', '--reconstruction-samples', type=int)
parser.add_argument('-l', '--latent-samples', type=int)
parser.add_argument('-g', '--generative-samples', type=int)
parser.add_argument('--gpus', type=int, nargs='+')
parser.add_argument('--save-figures-to')
parser.add_argument(
    '-m', '--model', choices=['ae', 'conv_ae', 'vae'], required=True)
options = parser.parse_args()

print(options)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
train = Dataset(x_train.astype(np.float32) / 255, y_train)
test = Dataset(x_test.astype(np.float32) / 255, y_test)

get_batch = BatchGenerator(train.images)
number_of_batches = len(train.images) // options.batch_size
image_shape = (32, 32, 3)

if options.model == 'ae':
    model = ae.AE(image_shape=image_shape, latent_size=32)
elif options.model == 'conv_ae':
    model = conv_ae.ConvAE(
        image_shape=image_shape, filter_sizes=[8, 8], latent_size=32)
elif options.model == 'vae':
    model = vae.VAE(
        image_shape=image_shape, filter_sizes=[32], latent_size=256)

model.compile(
    options.lr,
    decay_learning_rate_after=number_of_batches,
    learning_rate_decay=options.lr_decay)

trainer = trainer.Trainer(options.epochs, number_of_batches,
                          options.batch_size, options.gpus)
trainer.train(model, get_batch)

# Visualization Code

if options.save_figures_to is not None:
    visualize.disable_display()

if options.reconstruction_samples is not None:
    original_images = test.images[:options.reconstruction_samples]
    visualize.reconstructions(
        model, original_images, save_to=options.save_figures_to)

if options.latent_samples is not None:
    original_images = test.images[:options.latent_samples]
    labels = test.labels[:options.latent_samples]
    visualize.latent_space(
        model, original_images, labels, save_to=options.save_figures_to)

if options.generative_samples is not None:
    visualize.generative_samples(
        model, options.generative_samples, save_to=options.save_figures_to)

if options.save_figures_to is None:
    visualize.show()

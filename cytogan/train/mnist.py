#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials import mnist

from cytogan.extra import distributions, logs, misc
from cytogan.models import (ae, began, conv_ae, dcgan, infogan, lsgan, model,
                            vae, wgan)
from cytogan.train import common, trainer, visualize

parser = common.make_parser('cytogan-mnist')
options = common.parse_args(parser)

log = logs.get_root_logger(options.log_file)
log.debug('Options:\n%s', options.as_string)

if not options.show_figures:
    visualize.disable_display()

data = mnist.input_data.read_data_sets('MNIST_data', one_hot=False)
get_batch = lambda n: data.train.next_batch(n)[0].reshape([-1, 28, 28, 1])
number_of_batches = data.train.num_examples // options.batch_size
image_shape = (28, 28, 1)

learning = model.Learning(options.lr, options.lr_decay, options.lr_decay_steps
                          or number_of_batches)

if options.model == 'ae':
    hyper = ae.Hyper(image_shape, latent_size=32)
    Model = ae.AE
elif options.model == 'conv_ae':
    hyper = conv_ae.Hyper(image_shape, filter_sizes=(8, 8), latent_size=32)
    Model = conv_ae.ConvAE
elif options.model == 'vae':
    hyper = vae.Hyper(image_shape, filter_sizes=[32], latent_size=512)
    latent_distribution = lambda n: np.random.randn(n, hyper.latent_size)
    Model = vae.VAE
elif options.model in ('dcgan', 'lsgan', 'wgan'):
    hyper = dcgan.Hyper(
        image_shape,
        generator_filters=(128, 64, 32, 16),
        generator_strides=(1, 2, 2, 1),
        discriminator_filters=(128, 64, 32, 16),
        discriminator_strides=(1, 2, 2, 2),
        latent_size=100,
        noise_size=100,
        initial_shape=(7, 7))
    models = dict(dcgan=dcgan.DCGAN, lsgan=lsgan.LSGAN, wgan=wgan.WGAN)
    Model = models[options.model]
elif options.model == 'began':
    hyper = began.Hyper(
        image_shape,
        generator_filters=(128, 128, 128, 128),
        generator_strides=(1, 2, 2, 1),
        encoder_filters=(128, 256, 384),
        encoder_strides=(1, 2, 2, 1),
        decoder_filters=(128, 128, 128, 128),
        decoder_strides=(1, 2, 2, 1),
        latent_size=100,
        noise_size=100,
        initial_shape=(7, 7),
        diversity_factor=0.5,
        proportional_gain=1e-4)
    Model = began.BEGAN
elif options.model == 'infogan':
    latent_distribution = distributions.mixture({
        distributions.categorical(10):
        1,
        distributions.uniform():
        2,
    })
    hyper = infogan.Hyper(
        image_shape,
        generator_filters=(128, 64, 32, 16),
        generator_strides=(1, 2, 2, 1),
        discriminator_filters=(128, 64, 32, 16),
        discriminator_strides=(1, 2, 2, 2),
        latent_size=12,
        noise_size=100,
        initial_shape=(7, 7),
        latent_distribution=latent_distribution,
        discrete_variables=10,
        continuous_variables=2,
        continuous_lambda=0.8)
    Model = infogan.InfoGAN

log.debug('Hyperparameters:\n%s', misc.namedtuple_to_string(hyper))

trainer_options = trainer.Options(
    summary_directory=options.summary_dir,
    summary_frequency=options.summary_freq,
    checkpoint_directory=options.checkpoint_dir,
    checkpoint_frequency=options.checkpoint_freq)

trainer = trainer.Trainer(options.epochs, number_of_batches,
                          options.batch_size, trainer_options)

with common.get_session(options.gpus) as session:
    model = Model(hyper, learning, session)
    log.info('\n%s', model)
    if options.restore_from is None:
        session.run(tf.global_variables_initializer())
    else:
        model.restore(options.restore_from)
    if not options.skip_training:
        trainer.train(model, get_batch)

    if options.reconstruction_samples is not None:
        original_images, _ = data.test.next_batch(
            options.reconstruction_samples)
        original_images = original_images.reshape(-1, 28, 28, 1)
        visualize.reconstructions(
            model, original_images, gray=True, save_to=options.figure_dir)

    if options.latent_samples is not None:
        original_images, labels = data.test.next_batch(options.latent_samples)
        original_images = original_images.reshape(-1, 28, 28, 1)
        latent_vectors = model.encode(original_images)
        visualize.latent_space(
            latent_vectors, labels, save_to=options.figure_dir)

    if options.generative_samples is not None:
        if options.model == 'infogan':
            categorical = np.zeros([options.generative_samples, 10])
            categorical[:, 0] = 1
            continuous_1 = np.linspace(0, 1, options.generative_samples)
            continuous_2 = np.linspace(0, 1, options.generative_samples)
            samples = np.concatenate(
                [
                    categorical,
                    continuous_1.reshape(-1, 1),
                    continuous_2.reshape(-1, 1),
                ],
                axis=1)
        elif options.model.endswith('gan'):
            samples = np.random.randn(options.generative_samples,
                                      model.noise_size)
        else:
            samples = np.random.randn(options.generative_samples,
                                      model.latent_size)
        visualize.generative_samples(
            model, samples, gray=True, save_to=options.figure_dir)

if options.show_figures:
    visualize.show()

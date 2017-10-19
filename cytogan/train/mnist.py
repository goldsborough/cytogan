#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials import mnist

from cytogan.extra import distributions, logs, misc
from cytogan.models import (ae, began, conv_ae, dcgan, infogan, lsgan, model,
                            vae, wgan, orbital_gan)
from cytogan.train import common, trainer, visualize

parser = common.make_parser('cytogan-mnist')
options = common.parse_args(parser)

log = logs.get_root_logger(options.log_file)
log.debug('Options:\n%s', options.as_string)

if not options.show_figures:
    visualize.disable_display()

image_shape = (28, 28, 1)
data = mnist.input_data.read_data_sets('MNIST_data', one_hot=True)
number_of_batches = data.train.num_examples // options.batch_size
if options.conditional:
    conditional_shape = (10, )
    log.info('conditional shape: %d', conditional_shape[0])
else:
    conditional_shape = None


def get_batch(n):
    images, labels = data.train.next_batch(n)
    batch = [images.reshape((-1, ) + image_shape)]
    if options.with_labels:
        batch.append(labels.argmax(axis=1))
    if options.conditional:
        batch.append(labels)
    return batch[0] if len(batch) == 1 else batch


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
        initial_shape=(7, 7),
        conditional_shape=conditional_shape,
        conditional_embedding=None)
    models = dict(dcgan=dcgan.DCGAN, lsgan=lsgan.LSGAN, wgan=wgan.WGAN)
    Model = models[options.model]
elif options.model == 'began':
    hyper = began.Hyper(
        image_shape,
        generator_filters=(128, 128, 128),
        generator_strides=(1, 2, 2),
        encoder_filters=(128, 256, 384),
        encoder_strides=(1, 2, 2),
        decoder_filters=(128, 128, 128),
        decoder_strides=(1, 2, 2),
        latent_size=100,
        noise_size=100,
        initial_shape=(7, 7),
        diversity_factor=0.75,
        proportional_gain=1e-3,
        conditional_shape=conditional_shape)
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
        generator_filters=(256, 128, 64, 32),
        generator_strides=(1, 2, 2, 1),
        discriminator_filters=(32, 64, 128, 256),
        discriminator_strides=(1, 2, 2, 2),
        latent_size=12,
        noise_size=100,
        initial_shape=(7, 7),
        latent_distribution=latent_distribution,
        discrete_variables=10,
        continuous_variables=2,
        continuous_lambda=1,
        constrain_continuous=False,
        probability_loss='bce',
        continuous_loss='bce')
    Model = infogan.InfoGAN
elif options.model == 'ogan':
    hyper = orbital_gan.Hyper(
        image_shape,
        generator_filters=(128, 64, 32, 16),
        generator_strides=(1, 2, 2, 1),
        discriminator_filters=(128, 64, 32, 16),
        discriminator_strides=(1, 2, 2, 2),
        latent_size=100,
        noise_size=100,
        initial_shape=(7, 7),
        number_of_angles=10,
        number_of_radii=None,
        origin_label=0)
    Model = orbital_gan.OrbitalGAN

log.debug('Hyperparameters:\n%s', misc.namedtuple_to_string(hyper))

if options.workspace is not None and options.frames_per_epoch:
    save_every = number_of_batches // options.frames_per_epoch
    frame_options = trainer.FrameOptions(
        rate=common.Frequency(str(save_every)),
        sample=[np.random.randn(options.frame_sets, hyper.noise_size)],
        directory=options.frames_dir,
        number_of_sets=options.frame_sets)
else:
    frame_options = None

trainer_options = trainer.Options(
    summary_directory=options.summary_dir,
    summary_frequency=options.summary_freq,
    checkpoint_directory=options.checkpoint_dir,
    checkpoint_frequency=options.checkpoint_freq,
    frame_options=frame_options)

trainer = trainer.Trainer(options.epochs, number_of_batches,
                          options.batch_size, trainer_options)

if learning.decay:
    common.log_learning_rate_decay(options, learning, number_of_batches)

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
        labels = np.argmax(labels, axis=1)
        visualize.latent_space(
            latent_vectors, labels, save_to=options.figure_dir)

    if options.interpolate_samples is not None:
        start, end = np.random.randn(2, options.interpolate_samples[0],
                                     model.noise_size)
        if conditional_shape:
            labels = np.zeros([options.interpolate_samples[0], 10])
            for i in range(labels.shape[0]):
                labels[i, i] = 1
            labels = labels.repeat(options.interpolate_samples[1], axis=0)
            labels = np.array(list(labels))
        else:
            labels = None
        visualize.interpolation(
            model,
            start,
            end,
            *options.interpolate_samples,
            options.interpolation_method,
            conditional=labels,
            gray=True,
            save_to=options.figure_dir)

    if options.generative_samples is not None:
        if options.model == 'infogan':
            categorical = np.zeros([options.generative_samples, 10])
            half = options.generative_samples // 2
            categorical[:half, 0] = 1
            categorical[half:, 1] = 1
            continuous_1 = np.concatenate([
                np.linspace(-2, +2, options.generative_samples // 2),
                np.zeros(options.generative_samples // 2)
            ])
            continuous_2 = np.concatenate([
                np.zeros(options.generative_samples // 2),
                np.linspace(-2, +2, options.generative_samples // 2)
            ])
            latent = np.concatenate(
                [
                    categorical,
                    continuous_1.reshape(-1, 1),
                    continuous_2.reshape(-1, 1),
                ],
                axis=1)
            noise = np.random.randn(1, model.noise_size).repeat(
                options.generative_samples, axis=0)
            samples = [noise, latent]
        elif options.model.endswith('gan'):
            samples = np.random.randn(options.generative_samples,
                                      model.noise_size)
        else:
            samples = np.random.randn(options.generative_samples,
                                      model.latent_size)
        if conditional_shape:
            samples = [samples]
            variables = distributions.categorical(10)
            samples.append(variables((options.generative_samples)))
        visualize.generative_samples(
            model, samples, gray=True, save_to=options.figure_dir)

if options.show_figures:
    visualize.show()

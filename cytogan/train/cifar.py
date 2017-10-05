#!/usr/bin/env python3

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10

from cytogan.data.batch_generator import BatchGenerator
from cytogan.models import ae, conv_ae, model, vae, dcgan, infogan, began
from cytogan.train import common, trainer, visualize
from cytogan.train.common import Dataset, make_parser
from cytogan.extra import distributions, logs, misc

parser = make_parser('cytogan-cifar')
options = common.parse_args(parser)

log = logs.get_root_logger(options.log_file)
log.debug('Options:\n%s', options.as_string)

if not options.show_figures:
    visualize.disable_display()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
train = Dataset(x_train.astype(np.float32) / 255, y_train)
test = Dataset(x_test.astype(np.float32) / 255, y_test)

get_batch = BatchGenerator(train.images)
number_of_batches = len(train.images) // options.batch_size
image_shape = (32, 32, 3)

learning = model.Learning(options.lr, options.lr_decay, options.lr_decay_steps
                          or number_of_batches)

if options.model == 'ae':
    hyper = ae.Hyper(image_shape, latent_size=32)
    Model = ae.AE
elif options.model == 'conv_ae':
    hyper = conv_ae.Hyper(image_shape, filter_sizes=[8, 8], latent_size=32)
    Model = conv_ae.ConvAE
elif options.model == 'vae':
    hyper = vae.Hyper(
        image_shape, filter_sizes=[128, 128, 128], latent_size=512)
    Model = vae.VAE
elif options.model == 'dcgan':
    hyper = dcgan.Hyper(
        image_shape,
        generator_filters=(128, 64, 32, 16),
        generator_strides=(1, 2, 2, 1),
        discriminator_filters=(128, 64, 32, 16),
        discriminator_strides=(1, 2, 2, 1),
        latent_size=100,
        noise_size=100,
        initial_shape=(8, 8))
    Model = dcgan.DCGAN
elif options.model == 'began':
    hyper = began.Hyper(
        image_shape,
        generator_filters=(128, 128, 128, 128),
        generator_strides=(1, 1, 2, 2),
        encoder_filters=(128, 256, 384),
        encoder_strides=(1, 2, 2),
        decoder_filters=(128, 128, 128, 128),
        decoder_strides=(1, 1, 2, 2),
        latent_size=100,
        noise_size=100,
        initial_shape=(8, 8),
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
        discriminator_strides=(1, 2, 2, 1),
        latent_size=12,
        noise_size=100,
        initial_shape=(8, 8),
        latent_distribution=latent_distribution,
        discrete_variables=10,
        continuous_variables=2,
        continuous_lambda=0.8)
    Model = infogan.InfoGAN

log.debug('Hyperparameters:\n%s', misc.namedtuple_to_string(hyper))

trainer = trainer.Trainer(options.epochs, number_of_batches,
                          options.batch_size)
trainer.summary_directory = options.summary_dir
trainer.summary_frequency = options.summary_freq
trainer.checkpoint_directory = options.checkpoint_dir
trainer.checkpoint_frequency = options.checkpoint_freq
with common.get_session(options.gpus) as session:
    model = Model(hyper, learning, session)
    log.info('\n%s', model)
    if options.restore_from is None:
        tf.global_variables_initializer().run(session=session)
    else:
        model.restore(options.restore_from)
    if not options.skip_training:
        trainer.train(model, get_batch)

    if options.reconstruction_samples is not None:
        original_images = test.images[:options.reconstruction_samples]
        visualize.reconstructions(
            model, original_images, save_to=options.figure_dir)

    if options.latent_samples is not None:
        original_images = test.images[:options.latent_samples]
        labels = test.labels[:options.latent_samples].squeeze()
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
            model, samples, save_to=options.figure_dir)

if options.show_figures:
    visualize.show()

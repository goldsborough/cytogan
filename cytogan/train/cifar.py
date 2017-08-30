#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10

from cytogan.data.batch_generator import BatchGenerator
from cytogan.models import ae, conv_ae, model, vae
from cytogan.train import common, trainer, visualize
from cytogan.train.common import Dataset, make_parser
from cytogan.extra import logs

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
        labels = test.labels[:options.latent_samples]
        visualize.latent_space(
            model, original_images, labels, save_to=options.figure_dir)

    if options.generative_samples is not None:
        visualize.generative_samples(
            model, options.generative_samples, save_to=options.figure_dir)

if options.show_figures:
    visualize.show()

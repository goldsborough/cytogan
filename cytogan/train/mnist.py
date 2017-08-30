#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.examples.tutorials import mnist

from cytogan.models import ae, conv_ae, model, vae
from cytogan.train import common, trainer, visualize

parser = common.make_parser(name='cytogan-mnist')
options = parser.parse_args()

print(options)

if options.save_figures_to is not None:
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
    hyper = conv_ae.Hyper(image_shape, filter_sizes=[8, 8], latent_size=32)
    Model = conv_ae.ConvAE
elif options.model == 'vae':
    hyper = vae.Hyper(image_shape, filter_sizes=[32], latent_size=512)
    Model = vae.VAE

trainer = trainer.Trainer(options.epochs, number_of_batches,
                          options.batch_size)
trainer.summary_directory = options.summary_dir
trainer.summary_frequency = options.summary_freq
trainer.checkpoint_directory = options.checkpoint_dir
trainer.checkpoint_frequency = options.checkpoint_freq

with common.get_session(options.gpus) as session:
    model = Model(hyper, learning, session)
    print(model)
    if options.restore_from is None:
        tf.global_variables_initializer().run(session=session)
    else:
        model.restore(options.restore_from)
    if not options.skip_training:
        trainer.train(model, get_batch)

    if options.reconstruction_samples is not None:
        original_images, _ = data.test.next_batch(
            options.reconstruction_samples)
        original_images = original_images.reshape(-1, 28, 28, 1)
        visualize.reconstructions(
            model, original_images, gray=True, save_to=options.save_figures_to)

    if options.latent_samples is not None:
        original_images, labels = data.test.next_batch(options.latent_samples)
        original_images = original_images.reshape(-1, 28, 28, 1)
        visualize.latent_space(
            model, original_images, labels, save_to=options.save_figures_to)

    if options.generative_samples is not None:
        visualize.generative_samples(
            model,
            options.generative_samples,
            gray=True,
            save_to=options.save_figures_to)

if options.save_figures_to is None:
    visualize.show()

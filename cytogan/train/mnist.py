#!/usr/bin/env python3

from tensorflow.examples.tutorials import mnist

from cytogan.models import ae, conv_ae, vae
from cytogan.train import common, trainer, visualize

parser = common.make_parser(name='cytogan-mnist')
options = parser.parse_args()

print(options)

if options.save_figures_to is not None:
    visualize.disable_display()

data = mnist.input_data.read_data_sets('MNIST_data', one_hot=False)
get_batch = lambda n: data.train.next_batch(n)[0].reshape([-1, 28, 28, 1])
number_of_batches = data.train.num_examples // options.batch_size

if options.model == 'ae':
    model = ae.AE(image_shape=[28, 28, 1], latent_size=32)
elif options.model == 'conv_ae':
    model = conv_ae.ConvAE(
        image_shape=[28, 28, 1], filter_sizes=[8, 8], latent_size=32)
elif options.model == 'vae':
    model = vae.VAE(
        image_shape=[28, 28, 1], filter_sizes=[32, 32], latent_size=512)

model.compile(
    options.lr,
    decay_learning_rate_after=number_of_batches,
    learning_rate_decay=options.lr_decay)

print(model)

trainer = trainer.Trainer(options.epochs, number_of_batches,
                          options.batch_size)
trainer.summary_directory = options.summary_dir
trainer.summary_frequency = options.summary_freq
trainer.checkpoint_directory = options.checkpoint_dir
trainer.checkpoint_frequency = options.checkpoint_freq
with common.get_session(options.gpus) as session:
    trainer.train(session, model, get_batch)

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

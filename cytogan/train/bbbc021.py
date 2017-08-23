#!/usr/bin/env python3

import numpy as  np

from cytogan.models import ae, conv_ae, vae
from cytogan.train import trainer, visualize
from cytogan.data.cell_data import CellData
from cytogan.score import score_profiles

from cytogan.train import common

parser = common.make_parser('cytogan-bbbc021')
parser.add_argument('--metadata', required=True)
parser.add_argument('--labels', required=True)
parser.add_argument('--images', required=True)
parser.add_argument('-p', '--pattern', action='append')
options = parser.parse_args()
print(options)

# ['Week4_27481/G02', 'Week4_27521/B05']
cell_data = CellData(
    metadata_file_path=options.metadata,
    labels_file_path=options.labels,
    image_root=options.images,
    patterns=options.pattern)

print(cell_data.number_of_images)

b = cell_data.all_images()
k = list(b.keys())
profile = np.arange(100)
p = {}
for i in k:
    if cell_data.metadata.loc[i]['compound'] == 'anisomycin':
        p[i] = np.arange(100)
    else:
        p[i] = -np.arange(100)

d = cell_data.create_dataset_from_profiles(p)
c, accuracy = score_profiles(d)

visualize.confusion_matrix(c, title='MOA Confusion Matrix', accuracy=accuracy)
visualize.show()

print(c)
print(a)

# data = mnist.input_data.read_data_sets('MNIST_data', one_hot=False)
# get_batch = lambda n: data.train.next_batch(n)[0].reshape([-1, 28, 28, 1])
# number_of_batches = data.train.num_examples // options.batch_size
#
# if options.model == 'ae':
#     model = ae.AE(image_shape=[28, 28, 1], latent_size=32)
# elif options.model == 'conv_ae':
#     model = conv_ae.ConvAE(
#         image_shape=[28, 28, 1], filter_sizes=[8, 8], latent_size=32)
# elif options.model == 'vae':
#     model = vae.VAE(
#         image_shape=[28, 28, 1], filter_sizes=[32], latent_size=256)
#
# model.compile(
#     options.lr,
#     decay_learning_rate_after=number_of_batches,
#     learning_rate_decay=options.lr_decay)
#
# trainer = trainer.Trainer(options.epochs, number_of_batches,
#                           options.batch_size, options.gpus)
# trainer.train(model, get_batch)
#
# # Visualization Code
#
# if options.save_figures_to is not None:
#     visualize.disable_display()
#
# if options.reconstruction_samples is not None:
#     original_images, _ = data.test.next_batch(options.reconstruction_samples)
#     original_images = original_images.reshape(-1, 28, 28, 1)
#     visualize.reconstructions(
#         model, original_images, gray=True, save_to=options.save_figures_to)
#
# if options.latent_samples is not None:
#     original_images, labels = data.test.next_batch(options.latent_samples)
#     original_images = original_images.reshape(-1, 28, 28, 1)
#     visualize.latent_space(
#         model, original_images, labels, save_to=options.save_figures_to)
#
# if options.generative_samples is not None:
#     visualize.generative_samples(
#         model,
#         options.generative_samples,
#         gray=True,
#         save_to=options.save_figures_to)
#
# if options.save_figures_to is None:
#     visualize.show()

#!/usr/bin/env python3

import numpy as np

from cytogan.models import model, ae, conv_ae, vae
from cytogan.train import trainer, visualize
from cytogan.data.cell_data import CellData
from cytogan.metrics import profiling

from cytogan.train import common

parser = common.make_parser('cytogan-bbbc021')
parser.add_argument('--cell-count-file')
parser.add_argument('--metadata', required=True)
parser.add_argument('--labels', required=True)
parser.add_argument('--images', required=True)
parser.add_argument('-p', '--pattern', action='append')
parser.add_argument('--confusion', action='store_true')
options = parser.parse_args()
print(options)

if options.save_figures_to is not None:
    visualize.disable_display()

cell_data = CellData(
    metadata_file_path=options.metadata,
    labels_file_path=options.labels,
    image_root=options.images,
    cell_count_path=options.cell_count_file,
    patterns=options.pattern)

number_of_batches = cell_data.number_of_images // options.batch_size
image_shape = (128, 128, 3)

learning = model.Learning(options.lr, options.lr_decay, number_of_batches)

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
    trainer.train(model, cell_data.next_batch_of_images, options.restore_from)

    print('Evaluating ...')
    keys, images = cell_data.all_images()
    profiles = model.encode(images)
    outcome = cell_data.create_dataset_from_profiles(keys, profiles)
    confusion_matrix, accuracy = profiling.score_profiles(outcome)
    print('Final Accuracy: {0}'.format(accuracy))

    if options.confusion:
        visualize.confusion_matrix(
            confusion_matrix,
            title='MOA Confusion Matrix',
            accuracy=accuracy,
            save_to=options.save_figures_to)

    if options.reconstruction_samples is not None:
        visualize.reconstructions(
            model,
            images[:options.reconstruction_samples],
            save_to=options.save_figures_to)

    if options.latent_samples is not None:
        visualize.latent_space(
            model,
            images[:options.latent_samples],
            cell_data.labels.values,
            save_to=options.save_figures_to)

    if options.generative_samples is not None:
        visualize.generative_samples(
            model,
            options.generative_samples,
            gray=True,
            save_to=options.save_figures_to)

if options.save_figures_to is None:
    visualize.show()

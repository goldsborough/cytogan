#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from cytogan.data.cell_data import CellData
from cytogan.metrics import profiling
from cytogan.models import ae, conv_ae, model, vae
from cytogan.train import common, trainer, visualize

parser = common.make_parser('cytogan-bbbc021')
parser.add_argument('--cell-count-file')
parser.add_argument('--metadata', required=True)
parser.add_argument('--labels', required=True)
parser.add_argument('--images', required=True)
parser.add_argument('-p', '--pattern', action='append')
parser.add_argument('--confusion-matrix', action='store_true')
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
    tf.global_variables_initializer().run(session=session)
    if not options.skip_training:
        trainer.train(model, cell_data.next_batch, options.restore_from)

    print('Evaluating ...')
    keys, profiles = [], []
    batch_generator = cell_data.batches_of_size(options.batch_size)
    try:
        for batch_keys, images in tqdm(batch_generator, unit=' batches'):
            batch_profiles = model.encode(images)
            assert batch_profiles.shape == (options.batch_size,
                                            model.latent_size)
            profiles.append(batch_profiles)
            keys += batch_keys
    except KeyboardInterrupt:
        pass
    profiles = np.concatenate(profiles, axis=0)
    keys = list(cell_data.metadata.index)
    print('Generated {0} profiles ...'.format(len(profiles)))
    dataset = cell_data.create_dataset_from_profiles(keys, profiles)
    print('Matching {0:,} profiles to {1} MOAs ...'.format(
        len(dataset), len(dataset.moa.unique())))
    confusion_matrix, accuracy = profiling.score_profiles(dataset)
    print('Final Accuracy: {0:.3f}'.format(accuracy))

    if options.confusion_matrix:
        visualize.confusion_matrix(
            confusion_matrix,
            title='MOA Confusion Matrix',
            accuracy=accuracy,
            save_to=options.save_figures_to)

    if options.reconstruction_samples is not None:
        images = cell_data.next_batch(options.reconstruction_samples)
        visualize.reconstructions(
            model, np.stack(images, axis=0), save_to=options.save_figures_to)

    if options.latent_samples is not None:
        keys, images = cell_data.next_batch(
            options.latent_samples, with_keys=True)
        label_map, labels = cell_data.get_compound_indices(keys)
        visualize.latent_space(
            model, images, labels, label_map, save_to=options.save_figures_to)

    if options.generative_samples is not None:
        visualize.generative_samples(
            model,
            options.generative_samples,
            gray=True,
            save_to=options.save_figures_to)

if options.save_figures_to is None:
    visualize.show()

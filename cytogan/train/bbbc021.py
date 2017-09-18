#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from cytogan.data.cell_data import CellData
from cytogan.metrics import profiling
from cytogan.models import ae, conv_ae, model, vae, infogan, dcgan
from cytogan.train import common, trainer, visualize
from cytogan.extra import distributions, logs, misc

parser = common.make_parser('cytogan-bbbc021')
parser.add_argument('--cell-count-file')
parser.add_argument('--metadata', required=True)
parser.add_argument('--labels', required=True)
parser.add_argument('--images', required=True)
parser.add_argument('-p', '--pattern', action='append')
parser.add_argument('--confusion-matrix', action='store_true')
parser.add_argument('--latent-compounds', action='store_true')
parser.add_argument('--latent-moa', action='store_true')
options = common.parse_args(parser)
log = logs.get_root_logger(options.log_file)
log.debug('Options:\n%s', options.as_string)

if not options.show_figures:
    visualize.disable_display()

cell_data = CellData(
    metadata_file_path=options.metadata,
    labels_file_path=options.labels,
    image_root=options.images,
    cell_count_path=options.cell_count_file,
    patterns=options.pattern)

number_of_batches = cell_data.number_of_images // options.batch_size
image_shape = (96, 96, 3)

learning = model.Learning(options.lr, options.lr_decay, options.lr_decay_steps
                          or number_of_batches)

if options.model == 'ae':
    hyper = ae.Hyper(image_shape, latent_size=32)
    Model = ae.AE
elif options.model == 'conv_ae':
    hyper = conv_ae.Hyper(image_shape, filter_sizes=[8, 8], latent_size=32)
    Model = conv_ae.ConvAE
elif options.model == 'vae':
    hyper = vae.Hyper(image_shape, filter_sizes=[128, 64, 32], latent_size=256)
    Model = vae.VAE
elif options.model == 'dcgan':
    hyper = dcgan.Hyper(
        image_shape,
        generator_filters=(128, 64, 32, 16),
        generator_strides=(1, 2, 2, 2),
        discriminator_filters=(128, 64, 32, 16),
        discriminator_strides=(1, 2, 2, 2),
        latent_size=100,
        noise_size=100,
        initial_shape=(12, 12))
    Model = dcgan.DCGAN
elif options.model == 'infogan':
    discrete_variables = 32
    continuous_variables = 68
    latent_distribution = distributions.mixture({
        distributions.categorical(discrete_variables):
        1,
        distributions.uniform(-1.0, +1.0):
        continuous_variables,
    })
    hyper = infogan.Hyper(
        image_shape,
        generator_filters=(128, 64, 32, 16),
        generator_strides=(1, 2, 2, 2),
        discriminator_filters=(128, 64, 32, 16),
        discriminator_strides=(1, 2, 2, 2),
        latent_size=100,
        noise_size=100,
        initial_shape=(12, 12),
        latent_distribution=latent_distribution,
        discrete_variables=discrete_variables,
        continuous_variables=continuous_variables,
        continuous_lambda=1)
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
        trainer.train(model, cell_data.next_batch)

    log.info('Starting Evaluation')
    keys, profiles = [], []
    batch_generator = cell_data.batches_of_size(options.batch_size)
    try:
        for batch_keys, images in tqdm(batch_generator, unit=' batches'):
            profiles.append(model.encode(images))
            keys += batch_keys
    except KeyboardInterrupt:
        pass
    profiles = np.concatenate(profiles, axis=0)
    log.info('Generated %d profiles', len(profiles))
    dataset = cell_data.create_dataset_from_profiles(keys, profiles)
    log.info('Matching {0:,} profiles to {1} MOAs'.format(
        len(dataset), len(dataset.moa.unique())))
    treatment_profiles = profiling.reduce_profiles_across_treatments(dataset)
    log.info('Reduced dataset from %d to %d profiles for each treatment',
             len(dataset), len(treatment_profiles))
    confusion_matrix, accuracy = profiling.score_profiles(treatment_profiles)
    log.info('Final Accuracy: %.3f', accuracy)

    if options.confusion_matrix:
        visualize.confusion_matrix(
            confusion_matrix,
            title='MOA Confusion Matrix',
            accuracy=accuracy,
            save_to=options.figure_dir)

    if options.reconstruction_samples is not None:
        images = cell_data.next_batch(options.reconstruction_samples)
        visualize.reconstructions(
            model, np.stack(images, axis=0), save_to=options.figure_dir)

    if options.latent_samples is not None:
        keys, images = cell_data.next_batch(
            options.latent_samples, with_keys=True)
        treatment_names, indices = cell_data.get_treatment_indices(keys)
        latent_vectors = model.encode(images)
        visualize.latent_space(
            latent_vectors,
            indices,
            treatment_names,
            save_to=options.figure_dir,
            subject='Cells')

    if options.latent_compounds:
        compound_names, indices = cell_data.get_compound_indices(
            treatment_profiles)
        latent_vectors = np.array(list(treatment_profiles['profile']))
        visualize.latent_space(
            latent_vectors,
            indices,
            compound_names,
            save_to=options.figure_dir,
            subject='Compounds')

    if options.latent_moa:
        moa_names, indices = cell_data.get_moa_indices(treatment_profiles)
        latent_vectors = np.array(list(treatment_profiles['profile']))
        visualize.latent_space(
            latent_vectors,
            indices,
            moa_names,
            save_to=options.figure_dir,
            subject='MOA')

    if options.generative_samples is not None:
        if options.model == 'infogan':
            categorical = np.eye(options.generative_samples)
            categorical_zeros = np.zeros([
                options.generative_samples,
                discrete_variables - options.generative_samples
            ])
            continuous = np.linspace(-3, +3, options.generative_samples)
            continuous_zeros = np.zeros(
                [options.generative_samples, continuous_variables - 10])
            samples = np.concatenate(
                [categorical, categorical_zeros] +
                [continuous.reshape(-1, 1)] * 10 + [continuous_zeros],
                axis=1)
        elif options.model == 'dcgan':
            samples = np.random.randn(options.generative_samples,
                                      model.noise_size)
        else:
            samples = np.random.randn(options.generative_samples,
                                      model.latent_size)
        visualize.generative_samples(
            model, samples, save_to=options.figure_dir)

if options.show_figures:
    visualize.show()

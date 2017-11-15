#!/usr/bin/env python3

import os

import numpy as np
import scipy.misc
import tensorflow as tf
from tqdm import tqdm

from cytogan.data.cell_data import CellData
from cytogan.experiments import visualize, algebra, interpolation
from cytogan.extra import distributions, logs, misc
from cytogan.metrics import profiling
from cytogan.models import (ae, began, bigan, conv_ae, dcgan, infogan, lsgan,
                            model, vae, wgan)
from cytogan.train import common, trainer

parser = common.make_parser('cytogan-bbbc021')
parser.add_argument('--cell-count-file')
parser.add_argument('--concentration-only-labels', action='store_true')
parser.add_argument('--confusion-matrix', action='store_true')
parser.add_argument('--image-algebra-display-size', type=int, default=5)
parser.add_argument('--image-algebra-equations', type=int, default=1)
parser.add_argument('--image-algebra-sample-size', type=int, default=100)
parser.add_argument('--image-algebra', nargs='+', choices=algebra.EXPERIMENTS)
parser.add_argument('--images', required=True)
parser.add_argument('--interpolate-treatment-compound')
parser.add_argument(
    '--interpolate-treatment-concentrations', nargs='+', type=float)
parser.add_argument('--interpolate-treatment-length', type=int, default=10)
parser.add_argument('--interpolate-treatment-sample-size', type=int)
parser.add_argument('--interpolation-range', type=float, default=2.0)
parser.add_argument('--labels', required=True)
parser.add_argument('--latent-compounds', action='store_true')
parser.add_argument('--latent-concentrations', action='store_true')
parser.add_argument('--latent-moa', action='store_true')
parser.add_argument('--load-cell-data', action='store_true')
parser.add_argument('--load-profiles')
parser.add_argument('--load-treatment-profiles')
parser.add_argument('--metadata', required=True)
parser.add_argument('--no-latent-embedding', action='store_true')
parser.add_argument('--noise-file')
parser.add_argument('--normalize-luminance', action='store_true')
parser.add_argument('--save-profiles', action='store_true')
parser.add_argument('--save-generated-images', action='store_true')
parser.add_argument('--skip-evaluation', action='store_true')
parser.add_argument('--store-generated-noise', action='store_true')
parser.add_argument('--tsne-perplexity', type=int)
parser.add_argument('--vector-distance', action='store_true')
parser.add_argument('--whiten-profiles', action='store_true')
parser.add_argument('-p', '--pattern', action='append')
options = common.parse_args(parser)

if options.save_profiles:
    assert options.workspace is not None, 'Need workspace to store profiles'
    options.profiles_dir = os.path.join(options.workspace, 'profiles')
    if not os.path.exists(options.profiles_dir):
        os.makedirs(options.profiles_dir)

    def save_profiles(profiles, prefix):
        filename = '{0}.csv.gz'.format(prefix)
        log.info('Storing %s to disk', filename)
        path = os.path.join(options.profiles_dir, filename)
        profiling.save_profiles(path, profiles)


log = logs.get_root_logger(options.log_file)
log.debug('Options:\n%s', options.as_string)

if not options.show_figures:
    visualize.disable_display()

if not options.skip_evaluation or options.load_cell_data:
    cell_data = CellData(options.metadata, options.labels, options.images,
                         options.cell_count_file, options.pattern,
                         options.normalize_luminance, options.conditional,
                         options.concentration_only_labels)

image_shape = (96, 96, 3)
if options.skip_training:
    number_of_batches = 1
else:
    number_of_batches = cell_data.number_of_images // options.batch_size
if options.conditional:
    # one-hot encode the compound and have a
    # continuous variable for the concentration.
    conditional_shape = cell_data.label_shape
    log.info('conditional shape: %d', conditional_shape[0])
else:
    conditional_shape = None

learning = model.Learning(options.lr, options.lr_decay, options.lr_decay_steps
                          or number_of_batches)

embedding_size = None
if not (options.concentration_only_labels or options.no_latent_embedding):
    embedding_size = 16

if options.model == 'ae':
    hyper = ae.Hyper(image_shape, latent_size=32)
    Model = ae.AE
elif options.model == 'conv_ae':
    hyper = conv_ae.Hyper(image_shape, filter_sizes=[8, 8], latent_size=32)
    Model = conv_ae.ConvAE
elif options.model == 'vae':
    hyper = vae.Hyper(image_shape, filter_sizes=[128, 64, 32], latent_size=256)
    Model = vae.VAE
elif options.model in ('dcgan', 'lsgan', 'wgan'):
    hyper = dcgan.Hyper(
        image_shape,
        generator_filters=(256, 128, 64, 32),
        generator_strides=(1, 2, 2, 2),
        discriminator_filters=(32, 64, 128, 256),
        discriminator_strides=(1, 2, 2, 2),
        latent_size=100,
        noise_size=100,
        initial_shape=(12, 12),
        conditional_shape=conditional_shape,
        conditional_embedding=embedding_size,
        noise_kind='normal')
    models = dict(dcgan=dcgan.DCGAN, lsgan=lsgan.LSGAN, wgan=wgan.WGAN)
    Model = models[options.model]
elif options.model == 'began':
    hyper = began.Hyper(
        image_shape,
        generator_filters=(128, 128, 128, 128),
        generator_strides=(1, 2, 2, 2),
        encoder_filters=(64, 128, 256, 384),
        encoder_strides=(2, 2, 2, 2),
        decoder_filters=(128, 128, 128, 128),
        decoder_strides=(1, 2, 2, 2),
        latent_size=100,
        noise_size=100,
        initial_shape=(12, 12),
        diversity_factor=0.75,
        proportional_gain=1e-3,
        conditional_shape=conditional_shape,
        conditional_embedding=embedding_size,
        denoising=True)
    Model = began.BEGAN
elif options.model == 'infogan':
    discrete_variables = 0
    continuous_variables = 2
    latent_distribution = distributions.mixture({
        distributions.uniform(-1.0, +1.0):
        continuous_variables,
    })
    hyper = infogan.Hyper(
        image_shape,
        generator_filters=(256, 128, 64, 32),
        generator_strides=(1, 2, 2, 2),
        discriminator_filters=(32, 64, 128, 256),
        discriminator_strides=(1, 2, 2, 2),
        latent_size=discrete_variables + continuous_variables,
        noise_size=100,
        initial_shape=(12, 12),
        latent_distribution=latent_distribution,
        discrete_variables=discrete_variables,
        continuous_variables=continuous_variables,
        continuous_lambda=1,
        constrain_continuous=False,
        probability_loss='bce',
        continuous_loss='ll')
    Model = infogan.InfoGAN
elif options.model == 'bigan':
    hyper = bigan.Hyper(
        image_shape,
        generator_filters=(128, 64, 32, 16),
        generator_strides=(1, 2, 2, 2),
        encoder_filters=(128, 64, 32, 16),
        encoder_strides=(1, 2, 2, 2),
        discriminator_filters=[(128, 64, 32, 16), (1024, 1024, 256)],
        discriminator_strides=(1, 2, 2, 2),
        latent_size=100,
        initial_shape=(12, 12),
        noise_kind='uniform')
    Model = bigan.BiGAN

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

with common.get_session(options.gpus, options.random_seed) as session:
    model = Model(hyper, learning, session)
    log.info('\n%s', model)
    if options.restore_from is None:
        tf.global_variables_initializer().run(session=session)
    else:
        model.restore(options.restore_from)
    if not options.skip_training:
        trainer.train(model, cell_data.next_batch)

    if options.load_profiles:
        dataset = profiling.load_profiles(options.load_profiles, index=0)
        log.info('Found %s profiles', len(dataset))

    if options.load_treatment_profiles:
        treatment_profiles = profiling.load_profiles(
            options.load_treatment_profiles, index=0)
        log.info('Found %d treatment profiles', len(treatment_profiles))

    if not options.skip_evaluation:
        log.info('Starting Evaluation')

        loaded_profiles = (options.load_profiles
                           or options.load_treatment_profiles)
        if not loaded_profiles:
            keys, profiles = [], []
            batch_generator = cell_data.batches_of_size(options.batch_size)
            try:
                for batch_keys, images in tqdm(
                        batch_generator, unit=' batches'):
                    profiles.append(model.encode(images))
                    keys += batch_keys
            except KeyboardInterrupt:
                pass

            profiles = np.concatenate(profiles, axis=0)
            log.info('Generated %d profiles', len(profiles))

            dataset = cell_data.create_dataset_from_profiles(keys, profiles)
            log.info('Matching {0:,} profiles to {1} MOAs'.format(
                len(dataset), len(dataset.moa.unique())))

            if options.save_profiles:
                log.info('Storing profiles to disk')
                save_profiles(dataset, 'profiles')

        if options.whiten_profiles:
            profiling.whiten(dataset)
            log.info('Whitened data')
            if options.save_profiles:
                save_profiles(dataset, 'whitened')

        if not options.load_treatment_profiles:
            log.info('Collapsing profiles across treatments')
            treatment_profiles = profiling.reduce_profiles_across_treatments(
                dataset)
            log.info(
                'Reduced dataset from %d to %d profiles for each treatment',
                len(dataset), len(treatment_profiles))

            if options.save_profiles:
                save_profiles(treatment_profiles, 'treatments')

        # The DMSO (control) should not participate in the MOA classification.
        confusion_matrix, accuracy = profiling.score_profiles(
            treatment_profiles[treatment_profiles['compound'] != 'DMSO'])
        log.info('Final Accuracy: %.3f', accuracy)

        if options.confusion_matrix:
            visualize.confusion_matrix(
                confusion_matrix,
                title='MOA Confusion Matrix',
                accuracy=accuracy,
                save_to=options.figure_dir)

        if options.latent_compounds:
            _, indices = cell_data.get_compound_indices(treatment_profiles)
            latent_vectors = treatment_profiles['profile']
            point_sizes = treatment_profiles.groupby('compound').cumcount()
            visualize.latent_space(
                np.array(list(latent_vectors)),
                indices,
                point_sizes=np.array(list(point_sizes)),
                perplexity=options.tsne_perplexity,
                save_to=options.figure_dir,
                subject='Compounds')

        if options.latent_concentrations:
            _, indices = cell_data.get_concentration_indices(
                treatment_profiles)
            latent_vectors = treatment_profiles['profile']
            point_sizes = treatment_profiles.groupby('compound').cumcount()
            visualize.latent_space(
                np.array(list(latent_vectors)),
                indices,
                point_sizes=np.array(list(point_sizes)),
                perplexity=options.tsne_perplexity,
                save_to=options.figure_dir,
                subject='Concentrations')

        if options.latent_moa:
            moa_names, indices = cell_data.get_moa_indices(treatment_profiles)
            latent_vectors = treatment_profiles['profile']
            point_sizes = treatment_profiles.groupby('compound').cumcount()
            visualize.latent_space(
                np.array(list(latent_vectors)),
                indices,
                point_sizes=np.array(list(point_sizes)),
                perplexity=options.tsne_perplexity,
                save_to=options.figure_dir,
                subject='MOA',
                label_names=moa_names)

        if options.vector_distance:
            try:
                t = treatment_profiles
                start = t[t['concentration'] == 0.1]
                end = t[t['concentration'] == 0.3]
                intersection = set(start['compound']) & set(end['compound'])
                start = start[start['compound'].isin(intersection)]
                end = end[end['compound'].isin(intersection)]
                visualize.vector_distance(
                    np.array(list(start['profile'])),
                    np.array(list(end['profile'])),
                    labels=(0.1, 0.3),
                    save_to=options.figure_dir)
            except Exception as e:
                log.error('%s', e)

    # Kept here, but don't use it, it won't look good
    if options.latent_samples is not None:
        keys, images = cell_data.next_batch(
            options.latent_samples, with_keys=True)
        _, indices = cell_data.get_treatment_indices(keys)
        latent_vectors = model.encode(images)
        visualize.latent_space(
            latent_vectors,
            indices,
            perplexity=options.tsne_perplexity,
            save_to=options.figure_dir,
            subject='Cells')

    if options.reconstruction_samples is not None:
        image_pool = cell_data.next_batch(options.reconstruction_samples * 10)
        indices = np.random.randint(0,
                                    len(image_pool),
                                    options.reconstruction_samples)
        images = np.array(image_pool)[indices]
        assert len(images) == options.reconstruction_samples, len(images)
        visualize.reconstructions(
            model, np.stack(images, axis=0), save_to=options.figure_dir)

    if options.interpolation_samples is not None:
        assert options.interpolation_points >= 2
        points = np.random.randn(options.interpolation_points,
                                 options.interpolation_samples[0],
                                 model.noise_size)
        if conditional_shape:
            labels = cell_data.sample_labels(options.interpolation_samples[0])
            labels = np.array(list(labels))
            labels = labels.repeat(options.interpolation_samples[1], axis=0)
        else:
            labels = None
        visualize.interpolation(
            model,
            points,
            options.interpolation_samples[1],
            options.interpolation_method,
            options.save_interpolation_frames,
            options.interpolation_samples[0],
            conditional=labels,
            save_to=options.figure_dir)

    if options.interpolate_single_factors is not None:
        if options.interpolate_factors_from_images:
            start, end = interpolation.points_from_images(model, cell_data)
        else:
            start, end = np.random.randn(2, model.noise_size)
        visualize.single_factors(
            model,
            start,
            end,
            options.interpolate_single_factors[0],
            options.interpolate_single_factors[1],
            options.interpolation_method,
            save_to=options.figure_dir)

    if options.interpolate_treatment_compound is not None:
        assert options.interpolate_treatment_concentrations is not None
        points = interpolation.points_for_treatment(
            dataset,
            compound=options.interpolate_treatment_compound,
            concentrations=options.interpolate_treatment_concentrations,
            sample_size=options.interpolate_treatment_sample_size)
        visualize.interpolation(
            model,
            points,
            options.interpolate_treatment_length,
            options.interpolation_method,
            options.save_interpolation_frames,
            multi_point_interpolation_on_one_row=False,
            file_prefix='treatment-',
            save_to=options.figure_dir)

    if options.image_algebra:
        for experiment_name in options.image_algebra:
            experiment = algebra.get_experiment(
                experiment_name, options.image_algebra_equations)
            keys = experiment.keys(dataset, options.image_algebra_sample_size)
            images = cell_data.get_images(keys, in_order=True)
            assert len(images) == len(keys)
            lhs, rhs, base = np.split(np.array(images), 3, axis=0)
            vectors, result_images = experiment.calculate(
                model, lhs, rhs, base)
            result_vectors = np.split(vectors, 4, axis=0)[3]
            assert len(result_vectors) == len(result_images)
            labels = experiment.evaluate(result_vectors, treatment_profiles)
            visualize.image_algebra(
                model,
                lhs[:options.image_algebra_display_size],
                rhs[:options.image_algebra_display_size],
                base[:options.image_algebra_display_size],
                result_images[:options.image_algebra_display_size],
                labels=labels[:options.image_algebra_display_size],
                vectors=vectors,
                save_to=options.figure_dir)

    if options.generative_samples is not None:
        number_of_rows = None
        if options.model == 'infogan':
            latent = infogan.sample_variables(
                options.generative_samples, discrete_variables,
                continuous_variables, options.interpolation_range)
            noise = np.random.randn(1, model.noise_size).repeat(
                options.generative_samples, axis=0)
            samples = [noise, latent]
            number_of_rows = continuous_variables
        elif options.model.endswith('began'):
            samples = np.random.randn(options.generative_samples,
                                      model.latent_size)
        else:
            samples = np.random.randn(options.generative_samples,
                                      model.noise_size)

        if options.store_generated_noise:
            path = os.path.join(options.workspace, 'noise.csv')
            np.savetxt(path, samples, delimiter=',')

        # Fix two labels and sample random noise
        if conditional_shape:
            samples = [samples]
            labels = np.array(list(cell_data.sample_labels(2)))
            labels = labels.repeat(options.generative_samples // 2, axis=0)
            samples.append(labels)

        images = visualize.generative_samples(
            model,
            samples,
            save_to=options.figure_dir,
            number_of_rows=number_of_rows)

        # Fix noise and sample many labels (should look very different?)
        if conditional_shape:
            labels = cell_data.sample_labels(options.generative_samples // 2)
            labels = np.tile(np.array(list(labels)), (2, 1))
            noise = np.random.randn(2, model.noise_size)
            noise = np.repeat(noise, options.generative_samples // 2, axis=0)
            visualize.generative_samples(
                model, [noise, labels],
                save_to=options.figure_dir,
                filename='generative-samples2.png')

        if options.noise_file is not None:
            noise = np.loadtxt(options.noise_file, delimiter=',')
            if np.ndim(noise) == 1:
                noise = noise.reshape(1, -1)
            images = model.generate(noise)

        if options.save_generated_images:
            directory = os.path.join(options.figure_dir, 'generated')
            visualize.save_images(images, directory)

if options.show_figures:
    visualize.show()

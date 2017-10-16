import os
import time

import matplotlib.pyplot as plot
import numpy as np
import seaborn
import sklearn.manifold

from cytogan.extra import logs

log = logs.get_logger(__name__)

plot.style.use('ggplot')


def _plot_image_tile(number_of_rows, number_of_columns, index, image, gray):
    axis = plot.subplot(number_of_rows, number_of_columns, index + 1)
    plot.imshow(image, cmap=('gray' if gray else None))
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)


def _make_rgb(images):
    return images.repeat(3, axis=-1)


def _is_grayscale(images):
    return images.shape[-1] == 1


def _save_figure(folder, filename):
    path = os.path.join(folder, filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    log.info('Saving %s', path)
    plot.savefig(path, bbox_inches='tight', pad_inches=0)


def reconstructions(model,
                    original_images,
                    gray=False,
                    save_to=None,
                    title='Reconstructed Images'):
    reconstructed_images = model.reconstruct(original_images)
    if _is_grayscale(original_images):
        original_images = _make_rgb(original_images)
        reconstructed_images = _make_rgb(reconstructed_images)

    figure = plot.figure(figsize=(20, 4))
    figure.suptitle(title)
    number_of_images = len(original_images)
    for index in range(number_of_images):
        _plot_image_tile(2, number_of_images, index, original_images[index],
                         gray)
        _plot_image_tile(2, number_of_images, number_of_images + index,
                         reconstructed_images[index], gray)

    if save_to is not None:
        _save_figure(save_to, 'reconstructions.png')


def latent_space(latent_vectors,
                 labels=None,
                 perplexity=None,
                 point_sizes=None,
                 save_to=None,
                 subject=None):
    assert np.ndim(latent_vectors) == 2
    log.info('Plotting latent space for %d vectors', len(latent_vectors))

    if perplexity is None:
        perplexity = [2, 3, 4, 5] + list(range(10, 21)) + [30, 50, 70, 90]
    if isinstance(perplexity, int):
        perplexity = [perplexity]

    for p in perplexity:
        reduction = sklearn.manifold.TSNE(
            n_components=2, perplexity=p, init='pca', verbose=1)
        latent_vectors = reduction.fit_transform(latent_vectors)

        figure = plot.figure(figsize=(12, 10))
        subject_title = ' ({0})'.format(subject) if subject else ''
        subject_title += ' | P = {0}'.format(p)
        figure.suptitle('Latent Space{0}'.format(subject_title))
        plot.scatter(
            latent_vectors[:, 0],
            latent_vectors[:, 1],
            c=labels,
            lw=point_sizes,
            cmap=plot.cm.Spectral)

        if save_to is not None:
            subject_suffix = '-{0}'.format(subject.lower()) if subject else ''
            subject_suffix += '-{0}'.format(p)
            _save_figure(save_to, 'latent-space{0}.png'.format(subject_suffix))


def _linear_interpolation(start, end, number_of_samples):
    start, end = np.expand_dims(start, -1), np.expand_dims(end, -1)
    fractions = np.linspace(0, 1, number_of_samples)
    samples = (1 - fractions) * start + fractions * end
    return samples


def _slerp_interpolation(start, end, number_of_samples):
    # https://github.com/soumith/dcgan.torch/issues/14
    # Also: https://arxiv.org/pdf/1609.04468.pdf
    fractions = np.linspace(0, 1, number_of_samples)
    unit_start = start / np.linalg.norm(start)
    unit_end = end / np.linalg.norm(end)
    dot_products = np.sum(unit_start * unit_end, axis=1)
    omega = np.arccos(np.clip(dot_products, -1, 1)).reshape(-1, 1)
    omega_sine = np.sin(omega)

    start, end = np.expand_dims(start, -1), np.expand_dims(end, -1)
    if omega_sine.sum() == 0:
        return (1.0 - fractions) * start + fractions * end

    start_mix = np.sin((1.0 - fractions) * omega) / omega_sine
    end_mix = np.sin(fractions * omega) / omega_sine
    return np.expand_dims(start_mix, 1) * start + \
           np.expand_dims(end_mix, 1) * end


def interpolation(model,
                  start,
                  end,
                  number_of_interpolations,
                  interpolation_length,
                  method,
                  conditional=None,
                  gray=False,
                  save_to=None,
                  title='Latent Interpolation'):
    assert model.is_generative, model.name + ' is not generative'
    assert np.ndim(start) > 0, 'points must not be scalars'

    if method == 'linear':
        samples = _linear_interpolation(start, end, interpolation_length)
    elif method == 'slerp':
        samples = _slerp_interpolation(start, end, interpolation_length)

    k = number_of_interpolations
    split = [x.squeeze().T for x in np.split(samples, k)]
    samples = [np.concatenate(split, axis=0)]

    if conditional is not None:
        samples.append(conditional)

    images = model.generate(*samples).reshape(-1, *model.image_shape)

    if _is_grayscale(images):
        images = _make_rgb(images)

    figure = plot.figure(figsize=(8, k))
    # figure.suptitle(title)
    for index, image in enumerate(images):
        _plot_image_tile(k, interpolation_length, index, image, gray)

    plot.subplots_adjust(left=0.1, right=0.9, top=0.6, bottom=0.2)
    plot.tight_layout()

    if save_to is not None:
        filename = '{0}-interpolation.png'.format(method)
        _save_figure(save_to, filename)


def generative_samples(model,
                       samples,
                       gray=False,
                       save_to=None,
                       filename='generative-samples.png',
                       title='Generated Samples'):
    assert model.is_generative, model.name + ' is not generative'

    samples = samples if isinstance(samples, list) else [samples]
    images = model.generate(*samples).reshape(-1, *model.image_shape)
    if _is_grayscale(images):
        images = _make_rgb(images)
    figure = plot.figure(figsize=(10, 10))
    figure.suptitle(title)
    figure_rows = int(np.ceil(np.sqrt(len(images))))
    figure_columns = int(np.ceil(len(images) / figure_rows))
    for index, image in enumerate(images):
        _plot_image_tile(figure_rows, figure_columns, index, image, gray)

    if save_to is not None:
        _save_figure(save_to, filename)


def confusion_matrix(matrix,
                     title='Confusion Matrix',
                     accuracy=None,
                     save_to=None):
    figure, axis = plot.subplots(figsize=(14, 10))
    if accuracy:
        title += ' ({0:.1f}% Accuracy)'.format(accuracy * 100)
    figure.suptitle(title)
    seaborn.heatmap(matrix, annot=True, ax=axis)
    if save_to is not None:
        _save_figure(save_to, 'confusion-matrix.png')


def vector_distance(start,
                    end,
                    labels=None,
                    perplexity=15,
                    title=None,
                    save_to=None):
    tsne = sklearn.manifold.TSNE(
        n_components=2, perplexity=perplexity, init='pca', verbose=1)
    transformed = tsne.fit_transform(np.concatenate([start, end]))
    indices = np.tile(np.arange(len(start)), [2])

    figure, axis = plot.subplots(figsize=(5, 5))
    plot.scatter(
        transformed[:, 0], transformed[:, 1], c=indices, cmap='plasma')
    for x, y in zip(*np.split(transformed, 2)):
        delta = y - x
        plot.arrow(
            x[0],
            x[1],
            delta[0],
            delta[1],
            head_length=5,
            head_width=3,
            length_includes_head=True,
            color='r')

    if title is None:
        title = 'Vector Distance'
        if labels is not None:
            assert len(labels) == 2, labels
            title += ' between {0} and {1}'.format(*labels)
    figure.suptitle(title)

    if save_to is not None:
        _save_figure(save_to, 'vector-distance.png')


def disable_display():
    plot.switch_backend('Agg')


def show():
    plot.show()

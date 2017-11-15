import os

import matplotlib.pyplot as plot
import numpy as np
import seaborn
import sklearn.manifold
import scipy.misc

from cytogan.extra import logs

log = logs.get_logger(__name__)

plot.style.use('ggplot')


def _plot_image_tile(number_of_rows,
                     number_of_columns,
                     index,
                     image,
                     gray,
                     label=None):
    axis = plot.subplot(number_of_rows, number_of_columns, index + 1)
    plot.imshow(image, cmap=('gray' if gray else None))
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
    if label is not None:
        axis.text(0.05, -0.2, label, transform=axis.transAxes)


def _make_rgb(images):
    return images.repeat(3, axis=-1)


def _is_grayscale(images):
    return images.shape[-1] == 1


def _save_figure(folder, filename):
    path = os.path.join(folder, filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    log.info('Saving %s', path)
    plot.savefig(path, transparent=True)


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
                 subject=None,
                 label_names=None):
    assert np.ndim(latent_vectors) == 2
    log.info('Plotting latent space for %d vectors', len(latent_vectors))

    if perplexity is None:
        perplexity = [3, 5] + list(range(10, 21)) + [30, 50, 70, 90]
    if isinstance(perplexity, int):
        perplexity = [perplexity]

    log.info('Computing TSNEs at perplexity %s', tuple(perplexity))
    for p in perplexity:
        reduction = sklearn.manifold.TSNE(
            n_components=2, perplexity=p, init='pca', verbose=0)
        transformed_vectors = reduction.fit_transform(latent_vectors)

        figure = plot.figure(figsize=(12, 10))
        subject_title = ' ({0})'.format(subject) if subject else ''
        subject_title += ' | P = {0}'.format(p)
        figure.suptitle('Latent Space{0}'.format(subject_title))
        plot.scatter(
            transformed_vectors[:, 0],
            transformed_vectors[:, 1],
            c=labels,
            lw=point_sizes,
            cmap=plot.cm.Spectral)

        if label_names is not None:
            ticks = list(range(len(label_names)))
            colorbar = plot.colorbar(ticks=ticks)
            colorbar.ax.set_yticklabels(label_names)

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

    unit_start = start / np.linalg.norm(start, axis=-1).reshape(-1, 1)
    unit_end = end / np.linalg.norm(end, axis=-1).reshape(-1, 1)

    np.testing.assert_allclose(np.linalg.norm(unit_start, axis=-1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(unit_end, axis=-1), 1.0)

    dot_products = np.sum(unit_start * unit_end, axis=-1)

    omega = np.arccos(np.clip(dot_products, -1, 1)).reshape(-1, 1)
    omega_sine = np.sin(omega)

    start, end = np.expand_dims(start, -1), np.expand_dims(end, -1)
    if omega_sine.sum() == 0:
        return (1.0 - fractions) * start + fractions * end

    start_mix = np.sin((1.0 - fractions) * omega) / omega_sine
    end_mix = np.sin(fractions * omega) / omega_sine
    left = np.expand_dims(start_mix, 1) * start
    right = np.expand_dims(end_mix, 1) * end
    return left + right


def interpolation(model,
                  points,
                  interpolation_length,
                  method,
                  save_interpolation_frames=False,
                  number_of_interpolations=1,
                  conditional=None,
                  gray=False,
                  save_to=None,
                  file_prefix='',
                  multi_point_interpolation_on_one_row=True,
                  title='Latent Interpolation'):
    assert model.is_generative, model.name + ' is not generative'
    assert np.ndim(points[0]) > 0, 'points must not be scalars'
    assert method in ('linear', 'slerp'), method

    number_of_lines = len(points) - 1
    k = number_of_interpolations
    log.info('Interpolating between %d points', len(points))

    point_to_point = []
    for start, end in zip(points, points[1:]):
        if method == 'linear':
            samples = _linear_interpolation(start, end, interpolation_length)
        elif method == 'slerp':
            samples = _slerp_interpolation(start, end, interpolation_length)

        # Crazy rotation
        split = [x.squeeze().T for x in np.split(samples, k)]
        samples = [np.concatenate(split, axis=0)]

        if conditional is not None:
            samples.append(conditional)

        block = model.generate(*samples).reshape(-1, *model.image_shape)
        point_to_point.append(np.split(block, k, axis=0))

    images = [line[block] for block in range(k) for line in point_to_point]
    images = np.concatenate(images, axis=0)

    if _is_grayscale(images):
        images = _make_rgb(images)

    if save_interpolation_frames:
        assert save_to is not None
        for n, series in enumerate(np.split(images, k)):
            folder = os.path.join(save_to, 'interpolation', str(n))
            if not os.path.exists(folder):
                os.makedirs(folder)
            log.info('Storing interpolation frames to %s', folder)
            for i, image in enumerate(series):
                path = os.path.join(folder, '{0}.png'.format(i))
                scipy.misc.imsave(path, image)

    number_of_rows = k
    number_of_columns = interpolation_length
    if multi_point_interpolation_on_one_row:
        number_of_columns *= number_of_lines
    else:
        number_of_rows *= number_of_lines
    plot.figure(figsize=(10, 5))
    for index, image in enumerate(images):
        _plot_image_tile(number_of_rows, number_of_columns, index, image, gray)

    if save_to is not None:
        filename = '{0}{1}-interpolation.png'.format(file_prefix, method)
        _save_figure(save_to, filename)


def generative_samples(model,
                       samples,
                       gray=False,
                       save_to=None,
                       number_of_rows=None,
                       filename='generative-samples.png',
                       title='Generated Samples'):
    assert model.is_generative, model.name + ' is not generative'

    samples = samples if isinstance(samples, list) else [samples]
    images = model.generate(*samples).reshape(-1, *model.image_shape)
    if _is_grayscale(images):
        images = _make_rgb(images)
    if number_of_rows is None:
        number_of_rows = int(np.ceil(np.sqrt(len(images))))
    number_of_columns = int(np.ceil(len(images) / number_of_rows))
    figure = plot.figure(figsize=(10, min(10, number_of_rows)))
    figure.suptitle(title)
    for index, image in enumerate(images):
        _plot_image_tile(number_of_rows, number_of_columns, index, image, gray)

    if save_to is not None:
        _save_figure(save_to, filename)


def confusion_matrix(matrix,
                     title='Confusion Matrix',
                     accuracy=None,
                     save_to=None):
    figure, axis = plot.subplots(figsize=(14, 12))
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
        n_components=2, perplexity=perplexity, init='pca')
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


def single_factors(model,
                   start,
                   end,
                   factor_indices,
                   interpolation_length,
                   method,
                   gray=False,
                   save_to=None,
                   title='Single Factor Interpolation'):
    assert model.is_generative, model.name + ' is not generative'
    assert np.ndim(start) > 0, 'points must not be scalars'

    if isinstance(factor_indices, int):
        factor_indices = np.arange(factor_indices)

    if method == 'linear':
        interpolation = _linear_interpolation(start, end, interpolation_length)
    elif method == 'slerp':
        interpolation = _slerp_interpolation(start, end, interpolation_length)
        interpolation = interpolation.squeeze(axis=0)

    assert np.ndim(interpolation) == 2, interpolation.shape
    assert interpolation.shape[0] == len(start), interpolation.shape

    # Begin with only the start vector, tiled into three dimensions.
    repeats = (1, interpolation_length, len(factor_indices))
    base = np.tile(start.reshape(-1, 1, 1), repeats)

    # Insert the single rows of variation, one depth = one factor
    depths = np.arange(len(factor_indices))
    base[factor_indices, :, depths] = interpolation[factor_indices]
    np.testing.assert_allclose(base[factor_indices, -1, depths],
                               interpolation[factor_indices, -1])

    # Flatten out into a list of samples
    samples = base.T.reshape(-1, len(base))
    # Include the full interpolation in the first row
    samples = np.concatenate([interpolation.T, samples], axis=0)
    images = model.generate(samples).reshape(-1, *model.image_shape)

    if _is_grayscale(images):
        images = _make_rgb(images)

    number_of_rows = len(factor_indices) + 1
    plot.figure(figsize=(18, number_of_rows))
    for index, image in enumerate(images):
        _plot_image_tile(number_of_rows, interpolation_length, index, image,
                         gray)

    if save_to is not None:
        filename = '{0}-single-factors.png'.format(method)
        _save_figure(save_to, filename)


def subplot_equation(number_of_rows, row_index, lhs, rhs, base, result, labels,
                     gray):
    for n, image in enumerate([lhs, rhs, base, result]):
        index = (row_index * 4) + n
        _plot_image_tile(number_of_rows, 4, index,
                         image.squeeze(), gray, labels[n])


def image_algebra(model,
                  lhs,
                  rhs,
                  base,
                  result,
                  labels=None,
                  vectors=None,
                  gray=False,
                  save_to=None,
                  title='Image Algebra'):
    assert model.is_generative, model.name + ' is not generative'
    assert len(lhs) == len(rhs) == len(base) == len(result)

    if _is_grayscale(result):
        result = _make_rgb(result)

    number_of_equations = len(result)
    if labels is None:
        labels = [[None] * 4] * number_of_equations

    plot.figure(figsize=(7, 7))
    for n, equation in enumerate(zip(lhs, rhs, base, result, labels)):
        subplot_equation(number_of_equations, n, *equation, gray)

    if save_to is not None:
        _save_figure(save_to, 'image-algebra.png')

    if vectors is not None:
        vector_labels = np.repeat([0, 1, 2, 3], len(vectors) // 4)
        assert len(vector_labels) == len(vectors), (len(vector_labels),
                                                    len(vectors))
        transformed = sklearn.manifold.TSNE(
            init='pca', perplexity=30, verbose=1).fit_transform(vectors)
        plot.figure(figsize=(5, 5))
        plot.scatter(
            transformed[:, 0],
            transformed[:, 1],
            c=vector_labels,
            cmap='plasma')
        colorbar = plot.colorbar(ticks=[0, 1, 2, 3])
        colorbar.ax.set_yticklabels(['lhs', 'rhs', 'base', 'result'])
        if save_to is not None:
            _save_figure(save_to, 'image-algebra-vectors.png')

def save_images(images, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for n, image in enumerate(images):
        path = os.path.join(directory, '{0}.png'.format(n))
        scipy.misc.imsave(path, image)
    log.info('Saved %d images to %s', len(images), directory)


def disable_display():
    plot.switch_backend('Agg')


def show():
    log.info('Displaying figures')
    plot.show()

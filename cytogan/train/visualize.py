import os

import matplotlib
import matplotlib.pyplot as plot
import numpy as np
import scipy.stats
import sklearn.manifold
import seaborn

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
    print('Saving {0} ...'.format(path))
    plot.savefig(path)


def reconstructions(model, original_images, gray=False, save_to=None):
    reconstructed_images = model.reconstruct(original_images)
    if _is_grayscale(original_images):
        original_images = _make_rgb(original_images)
        reconstructed_images = _make_rgb(reconstructed_images)

    figure = plot.figure(figsize=(20, 4))
    figure.suptitle('Reconstructed Images')
    number_of_images = len(original_images)
    for index in range(number_of_images):
        _plot_image_tile(2, number_of_images, index, original_images[index],
                         gray)
        _plot_image_tile(2, number_of_images, number_of_images + index,
                         reconstructed_images[index], gray)

    if save_to is not None:
        _save_figure(save_to, 'reconstructions.png')


def latent_space(model,
                 images,
                 labels=None,
                 label_map=None,
                 reduction_method=sklearn.manifold.TSNE,
                 save_to=None):
    latent_vectors = model.encode(images)
    assert np.ndim(latent_vectors) == 2
    if latent_vectors.shape[1] > 2:
        print('Reducing dimensionality ...')
        reduction = reduction_method(n_components=2)
        latent_vectors = reduction.fit_transform(latent_vectors)
        assert latent_vectors.shape[1] == 2
    figure = plot.figure(figsize=(12, 10))
    figure.suptitle('Latent Space')
    plot.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=labels)
    if labels is not None:
        colorbar = plot.colorbar()
        if label_map is not None:
            ticks = {label_map[index] for index in sorted(set(labels))}
            colorbar.ax.set_yticklabels(ticks)

    if save_to is not None:
        _save_figure(save_to, 'latent-space.png')


def generative_samples(model,
                       number_of_samples,
                       distribution=np.random.randn,
                       gray=False,
                       save_to=None):
    samples = distribution(number_of_samples, model.latent_size)
    images = model.decode(samples).reshape(-1, *model.image_shape)
    if _is_grayscale(images):
        images = _make_rgb(images)
    figure = plot.figure(figsize=(10, 10))
    figure.suptitle('Decoded Samples')
    figure_rows = int(np.ceil(np.sqrt(number_of_samples)))
    figure_columns = int(np.ceil(number_of_samples / figure_rows))
    for index, image in enumerate(images[:number_of_samples]):
        _plot_image_tile(figure_rows, figure_columns, index, image, gray)

    if save_to is not None:
        _save_figure(save_to, 'generative-samples.png')


def confusion_matrix(matrix, title=None, accuracy=None, save_to=None):
    if title is not None:
        accuracy *= 100
        plot.suptitle('{0} ({1:.1f}% Accuracy)'.format(title, accuracy))
    _, ax = plot.subplots(figsize=(12, 10))
    seaborn.heatmap(matrix, annot=True, ax=ax)
    if save_to is not None:
        _save_figure(save_to, 'confusion-matrix.png')


def disable_display():
    plot.switch_backend('Agg')


def show():
    plot.show()

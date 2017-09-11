import os

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
    plot.savefig(path)


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
                 label_map=None,
                 reduction_method=sklearn.manifold.TSNE,
                 save_to=None,
                 subject=None):
    assert np.ndim(latent_vectors) == 2
    if latent_vectors.shape[1] > 2:
        log.info('Reducing dimensionality')
        reduction = reduction_method(n_components=2)
        latent_vectors = reduction.fit_transform(latent_vectors)
        assert latent_vectors.shape[1] == 2
    figure = plot.figure(figsize=(12, 10))
    subject_title = ' ({0})'.format(subject) if subject else ''
    figure.suptitle('Latent Space{0}'.format(subject_title))
    plot.scatter(
        latent_vectors[:, 0], latent_vectors[:, 1], c=labels, cmap='plasma')
    if labels is not None:
        colorbar = plot.colorbar()
        if label_map is not None:
            ticks = {label_map[index] for index in sorted(set(labels))}
            colorbar.ax.set_yticklabels(ticks)

    if save_to is not None:
        subject_suffix = '-{0}'.format(subject.lower()) if subject else ''
        _save_figure(save_to, 'latent-space{0}.png'.format(subject_suffix))


def generative_samples(model,
                       number_of_samples,
                       distribution=np.random.randn,
                       gray=False,
                       save_to=None,
                       title='Generated Samples'):
    samples = distribution(number_of_samples, model.latent_size)
    images = model.generate(samples).reshape(-1, *model.image_shape)
    if _is_grayscale(images):
        images = _make_rgb(images)
    figure = plot.figure(figsize=(10, 10))
    figure.suptitle(title)
    figure_rows = int(np.ceil(np.sqrt(number_of_samples)))
    figure_columns = int(np.ceil(number_of_samples / figure_rows))
    for index, image in enumerate(images[:number_of_samples]):
        _plot_image_tile(figure_rows, figure_columns, index, image, gray)

    if save_to is not None:
        _save_figure(save_to, 'generative-samples.png')


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


def disable_display():
    plot.switch_backend('Agg')


def show():
    plot.show()

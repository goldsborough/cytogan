import matplotlib.pyplot as plot
import numpy as np
import sklearn.manifold


def reconstructions(original_images, reconstructed_images, gray=False):
    if original_images.shape[-1] == 1:
        original_images = original_images.repeat(3, axis=-1)
        reconstructed_images = reconstructed_images.repeat(3, axis=-1)

    plot.figure(figsize=(20, 4))
    number_of_images = len(original_images)
    cmap = 'gray' if gray else None
    for index in range(number_of_images):
        axis = plot.subplot(2, number_of_images, index + 1)
        plot.imshow(original_images[index], cmap=cmap)
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

        axis = plot.subplot(2, number_of_images, number_of_images + index + 1)
        plot.imshow(reconstructed_images[index], cmap=cmap)
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)


def latent_space(latent_vectors,
                 labels,
                 reduction_method=sklearn.manifold.TSNE):
    assert np.ndim(latent_vectors) == 2
    if latent_vectors.shape[1] > 2:
        print('Reducing dimensionality ...')
        reduction = reduction_method(n_components=2)
        latent_vectors = reduction.fit_transform(latent_vectors)
        assert latent_vectors.shape[1] == 2
    plot.figure(figsize=(10, 10))
    plot.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=labels)
    plot.colorbar()


def show():
    plot.show()

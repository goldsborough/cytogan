import os.path
import scipy.misc
import collections


def _load_image(root_path, image_key, extension):
    full_path = os.path.join(root_path, '{0}.{1}'.format(image_key, extension))
    return scipy.misc.imread(full_path)


class LazyImageLoader(object):
    def __init__(self, root_path, extension='png'):
        self.root_path = root_path
        self.extension = extension
        self.loaded_images = {}

    def __getitem__(self, image_key):
        if isinstance(image_key, collections.Iterable):
            return self.get_all_images(image_key)
        return self.get_image(image_key)

    def get_image(self, image_key):
        image = self.loaded_images.get(image_key)
        if image is None:
            image = _load_image(self.root_path, image_key, self.extension)
            self.loaded_images[image_key] = image
        return image

    def get_all_images(self, image_keys):
        images = {}
        for key in image_keys:
            try:
                image = self.get_image(key)
            except Exception as error:
                print('Error loading image {0}: {1}'.format(key, repr(error)))
                continue
            images[key] = image

        return images

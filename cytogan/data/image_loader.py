import collections
import os.path

import scipy.misc
import numpy as np


def _load_image(root_path, image_key, extension):
    full_path = os.path.join(root_path, '{0}.{1}'.format(image_key, extension))
    image = scipy.misc.imread(full_path).astype(np.float32) / 255.
    if np.ndim(image) == 2:
        image = np.expand_dims(image, axis=-1)
    assert np.ndim(image) == 3
    return image


class LazyImageLoader(object):
    def __init__(self, root_path, extension='png', cache=True):
        self.root_path = root_path
        self.extension = extension
        self.do_cache = cache
        self.loaded_images = {}

    def __getitem__(self, image_key):
        if isinstance(image_key, collections.Iterable):
            return self.get_all_images(image_key)
        return self.get_image(image_key)

    def clear(self):
        self.loaded_images.clear()

    def get_image(self, image_key):
        if not self.do_cache:
            return _load_image(self.root_path, image_key, self.extension)
        image = self.loaded_images.get(image_key)
        if image is None:
            image = _load_image(self.root_path, image_key, self.extension)
            self.loaded_images[image_key] = image
        return image

    def get_all_images(self, image_keys):
        ok_keys = []
        ok_images = []
        for key in image_keys:
            try:
                image = self.get_image(key)
            except Exception as error:
                print('Error loading image {0}: {1}'.format(key, repr(error)))
            else:
                ok_keys.append(key)
                ok_images.append(image)

        return ok_keys, np.array(ok_images)

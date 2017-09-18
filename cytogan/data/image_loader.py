import collections
import multiprocessing
import os.path
import signal
import time

import numpy as np
import scipy.misc

from cytogan.extra import logs

log = logs.get_logger(__name__)


def load_image(root_path, image_key, extension):
    full_path = os.path.join(root_path, '{0}.{1}'.format(image_key, extension))
    image = scipy.misc.imread(full_path).astype(np.float32) / 255.
    # Expand a 2-D grayscale image into a 3-D image.
    if np.ndim(image) == 2:
        image = np.expand_dims(image, axis=-1)
    assert np.ndim(image) == 3
    return image


class AsyncImageLoader(object):
    '''Asynchronous image loader with prefetching function.'''

    class Job(object):
        '''Functor to circumvent limitations by multiprocessing.'''

        def __init__(self, root_path, extension):
            self.root_path = root_path
            self.extension = extension

        def __call__(self, key):
            # Ignore KeyboardInterrupt inside the worker processes.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            return load_image(self.root_path, key, self.extension)

    def __init__(self, root_path, extension='png'):
        self.futures = {}
        self.pool = multiprocessing.Pool()
        self.load_job = AsyncImageLoader.Job(root_path, extension)

    def __getitem__(self, image_keys):
        want = list(image_keys.copy())
        got_keys, got_images = [], []
        # We'll be iterating over `want` and removing elements from it during
        # iteration, so we cannot use a normal iterator or it will be
        # invalidated.
        index = 0
        # Note: this is a simple spin-loop. It asssumes that batches are
        # reasonably small and image loads reasonably fast so that we don't eat
        # up too much CPU spinning around. A more complete implementation would
        # use a condition variable, semaphore or a way of aggregating the
        # futures to make the calling thread actually block and sleep rather
        # than spin.
        while want:
            key = want[index]
            future = self.futures.get(key)
            if future is None:
                # Haven't started fetching this image yet at all.
                self.fetch_async([key])
                index += 1
            elif future.ready():
                # The value of the future is either the image or an exception.
                try:
                    image = future.get()
                    got_keys.append(key)
                    got_images.append(image)
                except IOError as error:
                    log.error(error)
                del want[index]
                # Free memory.
                del self.futures[key]
            else:
                # Job is still pending.
                index += 1
            # Wrap around.
            if index == len(want):
                index = 0
                time.sleep(0.005)

        return got_keys, got_images

    def fetch_async(self, image_keys):
        for key in image_keys:
            future = self.pool.apply_async(self.load_job, [key])
            self.futures[key] = future


class ImageLoader(object):
    '''A basic, synchronous image loader with caching functionality.'''

    def __init__(self, root_path, extension='png', cache=False):
        self.root_path = root_path
        self.extension = extension
        self.loaded_images = {}
        self.do_cache = cache

    def __getitem__(self, image_key):
        if isinstance(image_key, collections.Iterable):
            return self.get_all_images(image_key)
        return self.get_image(image_key)

    def clear(self):
        self.loaded_images.clear()

    def get_image(self, image_key):
        if not self.do_cache:
            return load_image(self.root_path, image_key, self.extension)
        image = self.loaded_images.get(image_key)
        if image is None:
            image = load_image(self.root_path, image_key, self.extension)
            self.loaded_images[image_key] = image
        return image

    def get_all_images(self, image_keys):
        ok_keys = []
        ok_images = []
        for key in image_keys:
            try:
                image = self.get_image(key)
            except IOError as error:
                log.error(
                    'Error loading image {0}: {1}'.format(key, repr(error)))
            else:
                ok_keys.append(key)
                ok_images.append(image)

        return ok_keys, ok_images

import collections
import os
import multiprocessing
import os.path
import time

import scipy.misc
import numpy as np


def load_image(key):
    image = scipy.misc.imread(key).astype(np.float32) / 255.
    print('Loaded ', key)
    return image


class ImageLoader(object):
    def __init__(self):
        self.pool = multiprocessing.Pool(multiprocessing.cpu_count())
        self.images = {}

    def __getitem__(self, image_keys):
        if isinstance(image_keys, str):
            want = [image_keys]
        else:
            want = image_keys[:]
        got_keys = []
        got_images = []
        index = 0
        while want:
            key = want[index]
            future = self.images.get(key)
            if future is None:
                self.fetch(key)
                index += 1
            elif future.ready():
                try:
                    image = future.get()
                    got_keys.append(key)
                    got_images.append(image)
                except IOError:
                    pass
                del want[index]
                del self.images[key]
            else:
                index += 1
            if index == len(want):
                index = 0

        return got_keys, got_images

    def fetch(self, image_keys):
        if isinstance(image_keys, str):
            image_keys = [image_keys]
        for key in image_keys:
            future = self.pool.apply_async(load_image, [key])
            self.images[key] = future


path = '../data/images/Week1_22123'
keys = [os.path.join(path, p) for p in os.listdir(path)][:1000]
loader = ImageLoader()
for start in range(0, len(keys), 100):
    k, i = loader[keys[start:start+100]]
    loader.fetch(keys[start+100:start+200])
    print('Processing ...')
    time.sleep(3)
    print('Done ...')

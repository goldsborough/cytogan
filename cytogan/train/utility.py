import collections
import numpy as np

Dataset = collections.namedtuple('Dataset', 'images, labels')

class BatchGenerator(object):
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __call__(self, batch_size):
        if self.index >= len(self.data):
            self.reset()

        stop_index = self.index + batch_size
        batch = self.data[self.index:stop_index]
        self.index = stop_index

        return batch

    def reset(self):
        self.index = 0
        np.random.shuffle(self.data)

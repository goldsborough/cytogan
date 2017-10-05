from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import numpy as np


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

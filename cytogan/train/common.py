import argparse
import collections
import os
import re
import time

import keras.backend as K
import numpy as np
import tensorflow as tf

Dataset = collections.namedtuple('Dataset', 'images, labels')


class Frequency(object):
    UNITS = dict(s=1, m=60, min=60, h=3600)

    def __init__(self, value):
        self.iterations = None
        self.seconds = None

        if isinstance(value, int):
            self.iterations = value
            return

        match = re.match(r'(\d+)\s*(s|m|min|h)?', value)
        if match is None:
            raise ValueError('{0} is not a valid frequency'.format(value))
        number = int(match.group(1))
        unit = match.group(2)
        if unit is None:
            self.iterations = number
        else:
            self.seconds = number * Frequency.UNITS[unit]

    def elapsed(self, number_of_iterations):
        if self.iterations is None:
            return int(time.time()) % self.seconds == 0
        return number_of_iterations % self.iterations == 0

    def __repr__(self):
        if self.iterations is None:
            return 'Frequency<{0} secs>'.format(self.seconds)
        return 'Frequency<{0} iter>'.format(self.iterations)


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


def make_parser(name):
    parser = argparse.ArgumentParser(description=name)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-decay', type=float)
    parser.add_argument('-r', '--reconstruction-samples', type=int)
    parser.add_argument('-l', '--latent-samples', type=int)
    parser.add_argument('-g', '--generative-samples', type=int)
    parser.add_argument('--gpus', type=int, nargs='+')
    parser.add_argument('--save-figures-to')
    parser.add_argument('--summary-dir')
    parser.add_argument(
        '--summary-freq', type=Frequency, default=Frequency(20))
    parser.add_argument('--checkpoint-dir')
    parser.add_argument(
        '--checkpoint-freq', type=Frequency, default=Frequency(20))
    parser.add_argument('--restore-from', metavar='CHECKPOINT_DIR')
    parser.add_argument(
        '-m', '--model', choices=['ae', 'conv_ae', 'vae'], required=True)

    return parser


def get_session(gpus):
    print('Using GPUs: {0}'.format(gpus))
    if gpus is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        gpus = ','.join(map(str, gpus))
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=gpus)

    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(session)

    return session

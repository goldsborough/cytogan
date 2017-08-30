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
        match = re.match(r'(\d+)\s*(s|m|min|h)?', value)
        if match is None:
            raise ValueError('{0} is not a valid frequency'.format(value))
        number = int(match.group(1))
        unit = match.group(2)
        if unit is None:
            self.iterations = number
        else:
            self.seconds = number * Frequency.UNITS[unit]
            self.last_check = 0

    def elapsed(self, number_of_iterations):
        if self.iterations is None:
            now = int(time.time())
            has_elapsed = (now - self.last_check) >= self.seconds
            self.last_check = now
            return has_elapsed
        return number_of_iterations % self.iterations == 0

    def __repr__(self):
        if self.iterations is None:
            return 'Frequency<{0} secs>'.format(self.seconds)
        return 'Frequency<{0} iter>'.format(self.iterations)


def make_parser(name):
    parser = argparse.ArgumentParser(description=name)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-decay', type=float)
    parser.add_argument('--lr-decay-steps', type=int)
    parser.add_argument('-r', '--reconstruction-samples', type=int)
    parser.add_argument('-l', '--latent-samples', type=int)
    parser.add_argument('-g', '--generative-samples', type=int)
    parser.add_argument('--gpus', type=int, nargs='+')
    parser.add_argument('--save-figures-to')
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--summary-dir')
    parser.add_argument(
        '--summary-freq', type=Frequency, default=Frequency('20'))
    parser.add_argument('--checkpoint-dir')
    parser.add_argument(
        '--checkpoint-freq', type=Frequency, default=Frequency('30s'))
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

import collections
import numpy as np
import argparse
import tensorflow as tf
import os

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


def make_parser(name):
    parser = argparse.ArgumentParser(description=name)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-decay', type=float, default=1)
    parser.add_argument('-r', '--reconstruction-samples', type=int)
    parser.add_argument('-l', '--latent-samples', type=int)
    parser.add_argument('-g', '--generative-samples', type=int)
    parser.add_argument('--gpus', type=int, nargs='+')
    parser.add_argument('--save-figures-to')
    parser.add_argument('--summary-dir')
    parser.add_argument('--summary-freq', default=20, type=int)
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
    tf.global_variables_initializer().run(session=session)

    return session

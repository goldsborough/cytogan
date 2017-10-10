import argparse
import collections
import os
import re
import sys
import time

import keras.backend as K
import tensorflow as tf

from cytogan.extra import logs
from cytogan import models

log = logs.get_logger(__name__)

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
            self.last_elapsed = 0

    def elapsed(self, number_of_iterations):
        if self.iterations is None:
            now = int(time.time())
            has_elapsed = (now - self.last_elapsed) >= self.seconds
            if has_elapsed:
                self.last_elapsed = now
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
    parser.add_argument('--lr', type=float, default=[1e-3], nargs='+')
    parser.add_argument('--lr-decay', type=float)
    parser.add_argument('--lr-decay-steps', type=int)
    parser.add_argument('-r', '--reconstruction-samples', type=int)
    parser.add_argument('-l', '--latent-samples', type=int)
    parser.add_argument('-g', '--generative-samples', type=int)
    parser.add_argument('-i', '--interpolate-samples', type=int, nargs=2)
    parser.add_argument(
        '--interpolation-method', default='linear', choices=('linear', 'slerp'))
    parser.add_argument('--gpus', type=int, nargs='+')
    parser.add_argument('--show-figures', action='store_true')
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument(
        '--summary-freq', type=Frequency, default=Frequency('20'))
    parser.add_argument(
        '--checkpoint-freq', type=Frequency, default=Frequency('30s'))
    parser.add_argument('--restore-from', metavar='CHECKPOINT_DIR')
    parser.add_argument('-w', '--workspace')
    parser.add_argument('-m', '--model', choices=models.MODELS, required=True)
    parser.add_argument('--dry', action='store_true')

    return parser


def parse_args(parser):
    options = parser.parse_args()

    if len(options.lr) == 1:
        options.lr = options.lr[0]

    options.checkpoint_dir = None
    options.summary_dir = None
    options.figure_dir = None
    options.log_file = None
    if options.workspace:
        timestamp = time.strftime('%d-%m-%Y_%H-%M-%S')
        run_name = '{0}_{1}'.format(options.model, timestamp)
        options.workspace = os.path.join(options.workspace, run_name)
        if not os.path.exists(options.workspace):
            os.makedirs(options.workspace)
        options.checkpoint_dir = os.path.join(options.workspace, 'checkpoints')
        options.summary_dir = os.path.join(options.workspace, 'summaries')
        options.figure_dir = os.path.join(options.workspace, 'figures')
        options.log_file = os.path.join(options.workspace, 'log.log')

    if options.model.startswith('c-'):
        options.model = options.model[2:]
        options.conditional = True
    else:
        options.conditional = False

    option_strings = ['{0} = {1}'.format(*i) for i in options._get_kwargs()]
    options.as_string = '\n'.join(option_strings)

    if options.dry:
        print(options.as_string)
        sys.exit()

    return options


def get_session(gpus):
    log.info('Using GPUs: %s', gpus)
    if gpus is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        gpus = ','.join(map(str, gpus))
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=gpus)

    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(session)
    K.manual_variable_initialization(True)

    tf.set_random_seed(42)

    return session


def log_learning_rate_decay(options, learning, number_of_batches):
    learning_rates = learning.rate
    if isinstance(learning.rate, float):
        learning_rates = [learning_rates]
    for index, lr in enumerate(learning_rates):
        steps = (number_of_batches / learning.steps_per_decay) * options.epochs
        final_learning_rate = lr * (learning.decay**steps)
        log.info('Learning rate %d will decay from %.5E to %.5E', index, lr,
                 final_learning_rate)

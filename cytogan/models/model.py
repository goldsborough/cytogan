import abc

import os
import time

import tensorflow as tf
import collections

from cytogan.extra import logs

log = logs.get_logger(__name__)

Learning = collections.namedtuple('Learning', 'rate, decay, steps_per_decay')


class Model(abc.ABC):
    def __init__(self, learning, session):
        assert isinstance(learning, Learning)

        self.session = session
        self.optimizer = None
        self._learning_rate = None

        # The training step indicator variable.
        self.global_step = tf.Variable(0, trainable=False)

        # Define the graph structure and setup the model architecture.
        self._define_graph()

        # Attach an optimizer and get the final learning rate tensor.
        self._add_optimizer(learning)

        # Boilerplate for management of the model execution.
        self._add_summaries()
        self.saver = tf.train.Saver(
            max_to_keep=2, keep_checkpoint_every_n_hours=6)

    @abc.abstractmethod
    def train_on_batch(self, batch, with_summary=False):
        pass

    @abc.abstractmethod
    def _define_graph(self):
        pass

    @abc.abstractmethod
    def _add_optimizer(self, learning, loss):
        pass

    @abc.abstractmethod
    def _add_summaries(self):
        pass

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def step(self):
        return self.global_step.eval(session=self.session)

    @property
    def learning_rate(self):
        assert not isinstance(self._learning_rate, collections.Iterable)
        if isinstance(self._learning_rate, float):
            return self._learning_rate
        return self._learning_rate.eval(session=self.session)

    @property
    def graph(self):
        return self.session.graph

    @property
    def is_generative(self):
        return hasattr(self, 'generate')

    def save(self, checkpoint_directory):
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        timestamp = time.strftime('%d-%m-%Y_%H-%M-%S')
        model_key = '{0}_{1}'.format(self.name, timestamp)
        checkpoint_path = os.path.join(checkpoint_directory, model_key)
        self.saver.save(
            self.session, checkpoint_path, global_step=self.global_step)

    def restore(self, checkpoint_path):
        if os.path.isdir(checkpoint_path):
            checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        else:
            checkpoint = checkpoint_path
        if checkpoint is None:
            raise RuntimeError(
                'Could not find any valid checkpoints under {0}!'.format(
                    checkpoint_path))
        log.info('Restoring from {0}'.format(checkpoint))
        self.saver.restore(self.session, checkpoint)

    def _get_learning_rate_tensor(self, initial_learning_rate, decay_rate,
                                  steps_per_decay):
        if decay_rate is None:
            return initial_learning_rate
        return tf.train.exponential_decay(
            initial_learning_rate,
            decay_steps=steps_per_decay,
            decay_rate=decay_rate,
            global_step=self.global_step,
            staircase=True)

import abc

import os
import time

import tensorflow as tf
import collections

from cytogan.extra import logs

log = logs.get_logger(__name__)

Learning = collections.namedtuple('Learning',
                                  'rate, decay, steps_per_decay, kwargs')


class Model(abc.ABC):
    def __init__(self, learning, session):
        self.session = session

        # The training step indicator variable.
        self.global_step = tf.Variable(0, trainable=False)

        # Define the graph structure and setup the model architecture.
        self.losses = self._define_graph()

        # Attach an optimizer and get the final learning rate tensor.
        self._learning_rates, self.optimizers = self._add_optimizers(
            learning, self.losses)

        # Boilerplate for management of the model execution.
        self._add_summaries()
        self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver(
            max_to_keep=10, keep_checkpoint_every_n_hours=1)

    @abc.abstractmethod
    def train_on_batch(self, batch, with_summary=False):
        ...

    @abc.abstractmethod
    def _define_graph(self):
        ...

    @abc.abstractmethod
    def _add_summaries(self):
        ...

    @property
    def name(self):
        return self.__class__.name__

    @property
    def step(self):
        return self.global_step.eval(session=self.session)

    @property
    def learning_rate(self):
        lr = list(self._learning_rates.values())[0]
        return lr if isinstance(lr, float) else lr.eval(session=self.session)

    @property
    def graph(self):
        return self.session.graph

    def save(self, checkpoint_directory):
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        timestamp = time.strftime('%d-%m-%Y_%H-%M-%S')
        model_key = '{0}_{1}'.format(self.name, timestamp)
        checkpoint_path = os.path.join(checkpoint_directory, model_key)
        self.saver.save(
            self.session, checkpoint_path, global_step=self.global_step)

    def restore(self, checkpoint):
        if os.path.isdir(checkpoint):
            checkpoint = tf.train.latest_checkpoint(checkpoint)
        log.info('Restoring from {0}'.format(checkpoint))
        if checkpoint is None:
            raise RuntimeError(
                'Could not find any valid checkpoints under {0}!'.format(
                    checkpoint))
        self.saver.restore(self.session, checkpoint)

    def _add_optimizers(self, learning_options, losses):
        learning_rates, optimizers = {}, {}
        if isinstance(losses, tf.Tensor):
            losses = {0: losses}
        if len(learning_options) < len(losses):
            learning_options = [learning_options[0]] * len(losses)
        assert len(losses) == len(learning_options)
        for learning, key in zip(learning_options, losses):
            learning_rate = self._get_learning_rate(learning)
            kwargs = learning.kwargs or {}
            optimizer = tf.train.AdamOptimizer(learning_rate, **kwargs)
            loss = tf.check_numerics(losses[key], key)
            optimizers[key] = optimizer.minimize(loss, self.global_step)
            learning_rates[key] = learning_rate

        return learning_rates, optimizers

    def _get_learning_rate(self, learning):
        # Start with the scalar learning rate value.
        if learning.decay is None:
            return learning.rate
        # Upgrade to decaying learning rate *tensor*.
        return tf.train.exponential_decay(
            learning.rate,
            decay_steps=learning.steps_per_decay,
            decay_rate=learning.decay,
            global_step=self.global_step,
            staircase=True)

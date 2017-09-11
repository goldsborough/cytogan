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
        assert isinstance(learning, Learning)

        self.session = session

        # The training step indicator variable.
        self.global_step = tf.Variable(0, trainable=False)

        # Define the graph structure and setup the model architecture.
        self.loss = self._define_graph()

        # Attach an optimizer and get the final learning rate tensor.
        tensors = self._add_optimizer(learning, self.loss)
        self._learning_rate, self.optimizer = tensors

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
        return self.__class__.__name__

    @property
    def step(self):
        return self.global_step.eval(session=self.session)

    @property
    def learning_rate(self):
        lr = self._learning_rate
        if isinstance(lr, dict):
            lr = list(lr.values())[0]
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

    def _add_optimizer(self, learning, losses):
        learning_rate, optimizer = {}, {}
        return_tensors = False
        if isinstance(losses, tf.Tensor):
            return_tensors = True
            losses = {0: losses}
        for index, key in enumerate(sorted(losses.keys())):
            lr = self._get_learning_rate(learning, index)
            kwargs = learning.kwargs or {}
            loss = tf.check_numerics(losses[key], str(key))
            variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=key) or None
            optimizer[key] = tf.train.AdamOptimizer(lr, **kwargs).minimize(
                loss, self.global_step, var_list=variables)
            learning_rate[key] = lr

        if return_tensors:
            return learning_rate[0], optimizer[0]
        return learning_rate, optimizer

    def _get_learning_rate(self, learning, index):
        learning_rate = learning.rate
        if isinstance(learning_rate, collections.Iterable):
            learning_rate = learning_rate[index]
        if learning.decay is None:
            return learning_rate
        decay = learning.decay
        if isinstance(decay, collections.Iterable):
            decay = decay[index]
        return tf.train.exponential_decay(
            learning_rate,
            decay_steps=learning.steps_per_decay,
            decay_rate=decay,
            global_step=self.global_step,
            staircase=True)

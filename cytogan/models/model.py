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
        self.session = session

        # The training step indicator variable.
        self.global_step = tf.Variable(0, trainable=False)

        # Define the graph structure and setup the model architecture.
        self.input, self.loss, self.model = self._define_graph()

        # Attach an optimizer and get the final learning rate tensor.
        self._learning_rate, self.optimize = self._add_optimizer(learning)

        # Boilerplate for management of the model execution.
        self._add_summaries()
        self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver(
            max_to_keep=10, keep_checkpoint_every_n_hours=1)

    def train_on_batch(self, batch, with_summary=False):
        fetches = [self.optimize, self.loss]
        if with_summary is not None:
            fetches.append(self.summary)
        outputs = self.session.run(fetches, feed_dict={self.input: batch})
        if with_summary:
            return outputs[1:]
        return outputs[1]

    @property
    def step(self):
        return self.global_step.eval(session=self.session)

    @property
    def learning_rate(self):
        if isinstance(self._learning_rate, float):
            return self._learning_rate
        return self._learning_rate.eval(session=self.session)

    @property
    def graph(self):
        return self.session.graph

    def save(self, checkpoint_directory):
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        class_name = self.__class__.__name__
        timestamp = time.strftime('%d-%m-%Y_%H-%M-%S')
        model_key = '{0}_{1}'.format(class_name, timestamp)
        checkpoint_path = os.path.join(checkpoint_directory, model_key)
        self.saver.save(
            self.session, checkpoint_path, global_step=self.global_step)

    def restore(self, checkpoint):
        if os.path.isdir(checkpoint):
            checkpoint = tf.train.latest_checkpoint(checkpoint)
        log.info('Restoring from {0} ...'.format(checkpoint))
        if checkpoint is None:
            raise RuntimeError(
                'Could not find any valid checkpoints under {0}!'.format(
                    checkpoint))
        self.saver.restore(self.session, checkpoint)

    @abc.abstractmethod
    def _define_graph(self):
        pass

    def _add_optimizer(self, learning):
        # Start with the scalar learning rate value.
        learning_rate = learning.rate
        if learning.decay is not None:
            # Upgrade to decaying learning rate *tensor*.
            learning_rate = tf.train.exponential_decay(
                learning.rate,
                decay_steps=learning.steps_per_decay,
                decay_rate=learning.decay,
                global_step=self.global_step,
                staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.loss, global_step=self.global_step)
        return learning_rate, optimizer

    def _add_summaries(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('learning_rate', self._learning_rate)

    def __repr__(self):
        lines = []
        try:
            # >= Keras 2.0.6
            self.model.summary(print_fn=lines.append)
        except TypeError:
            lines = [layer.name for layer in self.model.layers]
        return '\n'.join(map(str, lines))

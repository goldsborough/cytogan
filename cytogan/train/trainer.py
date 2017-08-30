import os
import time

import numpy as np
import tensorflow as tf
import tqdm

from cytogan.extra import logs

# Supress warnings about wrong compilation of TensorFlow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

log = logs.get_logger(__name__)


class Trainer(object):
    def __init__(self, number_of_epochs, number_of_batches, batch_size):
        self.number_of_epochs = number_of_epochs
        self.number_of_batches = number_of_batches
        self.batch_size = batch_size

        self.summary_directory = None
        self.summary_frequency = None
        self.summary_writer = None

        self.checkpoint_directory = None
        self.checkpoint_frequency = None

    def train(self, model, batch_generator):
        if self.summary_directory is not None:
            self.summary_writer = self._get_summary_writer(model.graph)

        start_time = time.time()
        try:
            self._train_loop(model, batch_generator)
        except KeyboardInterrupt:
            pass
        elapsed_time = time.time() - start_time

        if self.checkpoint_directory is not None:
            model.save(self.checkpoint_directory)

        log.info('Training complete! Took %.2fs', elapsed_time)

    def _train_loop(self, model, batch_generator):
        number_of_iterations = 0
        log_file = logs.LogFile(logs.get_raw_logger(__name__))
        for epoch_index in range(1, self.number_of_epochs + 1):
            batch_range = tqdm.trange(
                self.number_of_batches,
                unit=' batches',
                file=log_file,
                ncols=160)
            batch_range.set_description('Epoch {0}'.format(epoch_index))
            for _ in batch_range:
                lr = model.learning_rate
                batch = batch_generator(self.batch_size)
                if self._is_time_to_write_summary(number_of_iterations):
                    current_loss, summary = model.train_on_batch(
                        batch, with_summary=True)
                    self.summary_writer.add_summary(summary, model.step)
                else:
                    current_loss = model.train_on_batch(batch)
                if self._is_time_to_save_checkpoint(number_of_iterations):
                    model.save(self.checkpoint_directory)
                batch_range.set_postfix(loss=current_loss, lr=lr)
                if np.isnan(current_loss):
                    raise RuntimeError('Loss was NaN')
                number_of_iterations += 1

    def _is_time_to_write_summary(self, number_of_iterations):
        if self.summary_writer is not None:
            return self.summary_frequency.elapsed(number_of_iterations)
        return False

    def _is_time_to_save_checkpoint(self, number_of_iterations):
        if self.checkpoint_directory is not None:
            return self.checkpoint_frequency.elapsed(number_of_iterations)
        return False

    def _get_summary_writer(self, graph):
        log.info('Writing TensorBoard summaries to %s', self.summary_directory)
        return tf.summary.FileWriter(self.summary_directory, graph=graph)

    def __repr__(self):
        return 'Trainer<{0} epochs x {1} batches @ {2} examples>'.format(
            self.number_of_epochs, self.number_of_batches, self.batch_size)

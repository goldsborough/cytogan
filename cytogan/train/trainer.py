import collections
import os
import time

import numpy as np
import scipy.misc
import tensorflow as tf
import tqdm

from cytogan.extra import logs
from cytogan.extra.misc import namedtuple

FrameOptions = namedtuple('FrameOptions', [
    'rate',
    'sample',
    'directory',
    'number_of_sets',
])

Options = namedtuple('TrainerOptions', [
    'summary_directory',
    'summary_frequency',
    'checkpoint_directory',
    'checkpoint_frequency',
    'frame_options',
])

# Supress warnings about wrong compilation of TensorFlow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

log = logs.get_logger(__name__)


class Trainer(object):
    def __init__(self,
                 number_of_epochs,
                 number_of_batches,
                 batch_size,
                 options=Options()):
        self.number_of_epochs = number_of_epochs
        self.number_of_batches = number_of_batches
        self.batch_size = batch_size

        for index, field in enumerate(options._fields):
            setattr(self, field, options[index])
        self.summary_writer = None

    def train(self, model, batch_generator):
        if self.summary_directory is not None:
            self.summary_writer = self._get_summary_writer(model.graph)

        start_time = time.time()
        try:
            self._train_loop(model, batch_generator)
        except KeyboardInterrupt:
            print()
        elapsed_time = time.time() - start_time

        if self.checkpoint_directory is not None:
            model.save(self.checkpoint_directory)

        log.info('Training complete! Took %.2fs', elapsed_time)

    def _train_loop(self, model, batch_generator):
        log_file = logs.LogFile(logs.get_raw_logger(__name__))
        number_of_iterations = 0
        for epoch_index in range(1, self.number_of_epochs + 1):
            batch_range = self._get_batch_range(log_file, epoch_index)
            for _ in batch_range:
                batch = batch_generator(self.batch_size)
                if self._is_time_to_write_summary(number_of_iterations):
                    current_loss, summary = model.train_on_batch(
                        batch, with_summary=True)
                    if summary is not None:
                        self.summary_writer.add_summary(summary, model.step)
                else:
                    current_loss = model.train_on_batch(batch)
                if self._is_time_to_save_checkpoint(number_of_iterations):
                    model.save(self.checkpoint_directory)
                if self._is_time_to_generate_frame(number_of_iterations):
                    self._generate_frame(model)
                self._update_progressbar(batch_range, model.learning_rate,
                                         current_loss)
                number_of_iterations += 1

    def _is_time_to_write_summary(self, number_of_iterations):
        if self.summary_writer is not None:
            return self.summary_frequency.elapsed(number_of_iterations)
        return False

    def _is_time_to_save_checkpoint(self, number_of_iterations):
        if self.checkpoint_directory is not None:
            return self.checkpoint_frequency.elapsed(number_of_iterations)
        return False

    def _is_time_to_generate_frame(self, number_of_iterations):
        if self.frame_options is not None:
            return self.frame_options.rate.elapsed(number_of_iterations)
        return False

    def _get_summary_writer(self, graph):
        log.info('Writing TensorBoard summaries to %s', self.summary_directory)
        return tf.summary.FileWriter(self.summary_directory, graph=graph)

    def _get_batch_range(self, log_file, epoch_index):
        batch_range = tqdm.trange(
            self.number_of_batches, unit=' batches', file=log_file, ncols=160)
        batch_range.set_description('Epoch {0}'.format(epoch_index))

        return batch_range

    def _update_progressbar(self, batch_range, learning_rate, loss):
        if isinstance(loss, collections.Mapping):
            strings = {}
            for key, value in loss.items():
                assert np.isscalar(value), (key, value)
                strings[key] = '{0:.8f}'.format(value)
            for key, value in learning_rate.items():
                assert np.isscalar(value), (key, value)
                strings['lr_{0}'.format(key)] = '{0:.6f}'.format(value)
            batch_range.set_postfix(**strings)
        else:
            assert np.isscalar(loss), loss
            batch_range.set_postfix(loss=loss)

    def _generate_frame(self, model):
        assert model.is_generative, model.name + ' is not generative'
        frames = model.generate(*self.frame_options.sample)
        if not os.path.exists(self.frame_options.directory):
            os.makedirs(self.frame_options.directory)
            for i in range(self.frame_options.number_of_sets):
                os.makedirs(os.path.join(self.frame_options.directory, str(i)))
        frames = (frames * 255).astype(np.uint8)
        for i in range(self.frame_options.number_of_sets):
            directory = os.path.join(self.frame_options.directory, str(i))
            index = len(os.listdir(directory))
            filename = '{0}.png'.format(index)
            path = os.path.join(directory, filename)
            scipy.misc.imsave(path, frames[i].squeeze())

    def __repr__(self):
        return 'Trainer<{0} epochs x {1} batches @ {2} examples>'.format(
            self.number_of_epochs, self.number_of_batches, self.batch_size)

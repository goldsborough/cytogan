import os
import time

import keras
import numpy as np
import tensorflow as tf
import tqdm

# Supress warnings about wrong compilation of TensorFlow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Trainer(object):
    def __init__(self,
                 number_of_epochs,
                 number_of_batches,
                 batch_size,
                 summary_directory=None,
                 summary_frequency=None):
        self.number_of_epochs = number_of_epochs
        self.number_of_batches = number_of_batches
        self.batch_size = batch_size

        self.summary_directory = summary_directory
        self.summary_frequency = summary_frequency
        self.summary_writer = None

    def train(self, session, model, batch_generator):
        model.session = session
        if self.summary_directory is not None:
            self.summary_writer = tf.summary.FileWriter(
                self.summary_directory, graph=session.graph)
            print('Writing TensorBoard summaries to {0}'.format(
                self.summary_directory))
        start_time = time.time()
        try:
            self._train_loop(model, batch_generator)
        except KeyboardInterrupt:
            pass

        elapsed_time = time.time() - start_time
        print('Training complete! Took {0:.2f}s'.format(elapsed_time))

    def _train_loop(self, model, batch_generator):
        number_of_iterations = 0
        for epoch_index in range(1, self.number_of_epochs + 1):
            batch_range = tqdm.trange(self.number_of_batches, unit=' batches')
            batch_range.set_description('Epoch {0}'.format(epoch_index))
            for _ in batch_range:
                batch = batch_generator(self.batch_size)
                if (self.summary_writer and \
                    number_of_iterations % self.summary_frequency == 0):
                    current_loss = model.train_on_batch(
                        batch, self.summary_writer)
                else:
                    current_loss = model.train_on_batch(batch)
                batch_range.set_postfix(loss=current_loss)
                if np.isnan(current_loss):
                    raise RuntimeError('Loss was NaN')
                number_of_iterations += 1

    def __repr__(self):
        return 'Trainer<{0} epochs x {1} batches @ {2} examples>'.format(
            self.number_of_epochs, self.number_of_batches, self.batch_size)

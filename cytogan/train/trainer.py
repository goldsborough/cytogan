import time

import numpy as np
import tensorflow as tf
import tqdm


class Trainer(object):
    def __init__(self, number_of_epochs, number_of_batches, batch_size):
        self.number_of_epochs = number_of_epochs
        self.number_of_batches = number_of_batches
        self.batch_size = batch_size

    def train(self, model, session, batch_generator):
        start_time = time.time()
        tf.global_variables_initializer().run(session=session)
        try:
            self._train_loop(model, session, batch_generator)
        except KeyboardInterrupt:
            print()

        elapsed_time = time.time() - start_time
        print('Training complete! Took {0:.2f}s'.format(elapsed_time))

    def _train_loop(self, model, session, batch_generator):
        for epoch_index in range(1, self.number_of_epochs + 1):
            batch_range = tqdm.trange(self.number_of_batches)
            batch_range.set_description('Epoch {0}'.format(epoch_index))
            batch_range.set_postfix(loss=0)
            for batch_index in batch_range:
                batch = batch_generator(self.batch_size)
                current_loss = model.train_on_batch(batch)
                batch_range.set_postfix(loss=current_loss)
            if np.isnan(current_loss):
                raise RuntimeError('Loss was NaN')

    def __repr__(self):
        return 'Trainer<{0} epochs x {1} batches @ {2} examples>'.format(
            self.number_of_epochs, self.number_of_batches, self.batch_size)

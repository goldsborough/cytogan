import os
import time

import keras
import numpy as np
import tensorflow as tf
import tqdm

# Supress warnings about wrong compilation of TensorFlow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def use_gpus(visible_devices):
    gpu_options = tf.GPUOptions(
        allow_growth=True, visible_device_list=visible_devices)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    keras.backend.set_session(session)


class Trainer(object):
    def __init__(self,
                 number_of_epochs,
                 number_of_batches,
                 batch_size,
                 gpus=None):
        self.number_of_epochs = number_of_epochs
        self.number_of_batches = number_of_batches
        self.batch_size = batch_size

        print('Using GPUs: {0}'.format(gpus))
        if gpus is None:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        else:
            use_gpus(','.join(map(str, gpus)))

    def train(self, model, batch_generator):
        start_time = time.time()
        try:
            self._train_loop(model, batch_generator)
        except KeyboardInterrupt:
            pass

        elapsed_time = time.time() - start_time
        print('Training complete! Took {0:.2f}s'.format(elapsed_time))

    def _train_loop(self, model, batch_generator):
        for epoch_index in range(1, self.number_of_epochs + 1):
            batch_range = tqdm.trange(self.number_of_batches, unit=' batches')
            batch_range.set_description('Epoch {0}'.format(epoch_index))
            for _ in batch_range:
                batch = batch_generator(self.batch_size)
                current_loss = model.train_on_batch(batch)
                batch_range.set_postfix(
                    loss=current_loss, lr=model.learning_rate)
                if np.isnan(current_loss):
                    raise RuntimeError('Loss was NaN')

    def __repr__(self):
        return 'Trainer<{0} epochs x {1} batches @ {2} examples>'.format(
            self.number_of_epochs, self.number_of_batches, self.batch_size)

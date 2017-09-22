import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer


class BatchNorm(Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BatchNorm, self).build(input_shape)

    def call(self, tensor):
        with K.name_scope('batch_norm'):
            return tf.layers.batch_normalization(
                tensor,
                axis=-1,
                momentum=0.9,
                training=K.learning_phase(),
                fused=None)

    def compute_output_shape(self, input_shape):
        return input_shape


class UpSamplingNN(Layer):
    def __init__(self, factor, **kwargs):
        self.factor = factor
        super(UpSamplingNN, self).__init__(**kwargs)

    def build(self, input_shape):
        super(UpSamplingNN, self).build(input_shape)

    def call(self, images):
        with K.name_scope('up_sampling_nn'):
            new_height, new_width = self._scale(images.shape[1:-1])
            return tf.image.resize_nearest_neighbor(
                images, size=(new_height, new_width))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * self.factor,
                input_shape[2] * self.factor, input_shape[3])

    def _scale(self, sizes):
        return [self.factor * int(size) for size in sizes]


class AddNoise(Layer):
    def __init__(self, **kwargs):
        # github.com/soumith/ganhacks#13-add-noise-to-inputs-decay-over-time
        super(AddNoise, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AddNoise, self).build(input_shape)

    def call(self, images):
        with K.name_scope('add_noise'):
            return images + tf.random_normal(
                tf.shape(images), mean=0.0, stddev=0.1)

    def compute_output_shape(self, input_shape):
        return input_shape


class RandomNormal(Layer):
    def __init__(self, noise_size, mean=0.0, stddev=1.0, **kwargs):
        self.noise_size = noise_size
        self.mean = mean
        self.stddev = stddev
        super(RandomNormal, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RandomNormal, self).build(input_shape)

    def call(self, batch_size):
        with K.name_scope('random_normal'):
            batch_size = tf.squeeze(tf.cast(batch_size, tf.int32))
            return tf.random_normal(
                shape=(batch_size, self.noise_size),
                mean=self.mean,
                stddev=self.stddev)

    def compute_output_shape(self, input_shape):
        return (None, self.noise_size)

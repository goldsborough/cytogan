import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Concatenate, Dense, LeakyReLU, Reshape


class BatchNorm(Layer):
    def __init__(self, axis=-1, momentum=0.9, variance_epsilon=1e-3, **kwargs):
        self.axis = axis
        self.momentum = momentum
        self.variance_epsilon = variance_epsilon

        self.scale = None
        self.offset = None
        self.population_mean = None
        self.population_variance = None

        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = (input_shape[self.axis], )
        with K.name_scope('batch_norm'):
            self.scale = self.add_weight(
                shape=shape, name='scale', initializer='ones')
            self.offset = self.add_weight(
                shape=shape, name='offset', initializer='zeros')

            self.population_mean = self.add_weight(
                shape=shape,
                name='population_mean',
                initializer='zeros',
                trainable=False)
            self.population_variance = self.add_weight(
                shape=shape,
                name='population_variance',
                initializer='ones',
                trainable=False)

        super(BatchNorm, self).build(input_shape)

    def call(self, inputs):
        with K.name_scope('batch_norm'):
            return tf.cond(K.learning_phase(),
                           lambda: self._training_graph(inputs),
                           lambda: self._test_graph(inputs))

    def compute_output_shape(self, input_shape):
        return input_shape

    def _training_graph(self, inputs):
        reduction_axes = list(range(len(inputs.shape)))
        reduction_axes.pop(self.axis)

        mean, variance = tf.nn.moments(
            inputs, axes=reduction_axes, name='moments')
        variance = tf.maximum(variance, tf.constant(0.0))

        updated_mean = self._mix(self.population_mean, mean)
        update_population_mean = tf.assign(self.population_mean, updated_mean)

        updated_variance = self._mix(self.population_variance, variance)
        update_population_variance = tf.assign(self.population_variance,
                                               updated_variance)

        with tf.control_dependencies(
            [update_population_mean, update_population_variance]):
            return tf.nn.batch_normalization(
                inputs,
                mean,
                variance,
                self.offset,
                self.scale,
                variance_epsilon=self.variance_epsilon)

    def _test_graph(self, inputs):
        return tf.nn.batch_normalization(
            inputs,
            self.population_mean,
            self.population_variance,
            self.offset,
            self.scale,
            variance_epsilon=self.variance_epsilon)

    def _mix(self, old, new):
        return self.momentum * old + (1 - self.momentum) * new


class BatchNorm2(Layer):
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


def MixImagesWithVariables(images, variables):
    image_shape = list(map(int, images.shape[1:]))
    flat_size = np.prod(image_shape)
    new_shape = image_shape[:-1] + [image_shape[-1] * 2]

    with K.name_scope('mix_images_w_vars'):
        flat_images = Reshape([flat_size])(images)
        vectors = Concatenate(axis=1)([flat_images, variables])
        mix = Dense(flat_size * 2)(vectors)
        mix = LeakyReLU(alpha=0.2)(mix)
        volume = Reshape(new_shape)(mix)

    return volume

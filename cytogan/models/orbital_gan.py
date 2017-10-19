# 1. For each compound in batch:
#   1. Advance moving average
#   2. Penalize cosine distance between points of one compound and average
#   3. Find two nearest neighbors, slightly penalize closeness
# 2. For each concentration in batch:
#   1. Advance moving average
#   2. Penalize error in norm of one concentration and average
#   3. Find two nearest neighbors, slightly penalize closeness

import collections

import keras.backend as K
import keras.losses
import numpy as np
import tensorflow as tf
from keras.layers import Input

from cytogan.metrics import losses
from cytogan.models import lsgan, util

tf.logging.set_verbosity(tf.logging.INFO)

Hyper = collections.namedtuple('Hyper', [
    'image_shape',
    'generator_filters',
    'discriminator_filters',
    'generator_strides',
    'discriminator_strides',
    'latent_size',
    'noise_size',
    'initial_shape',
    'number_of_angles',
    'number_of_radii',
    'origin_label',
])


class OrbitalGAN(lsgan.LSGAN):
    def __init__(self, hyper, learning, session):
        super(OrbitalGAN, self).__init__(hyper, learning, session)

    def _train_discriminator(self, fake_images, real_images, labels,
                             with_summary):
        discriminator_labels = np.concatenate(
            [np.zeros(len(fake_images)),
             np.ones(len(real_images))], axis=0)
        images = np.concatenate([fake_images, real_images], axis=0)
        fetches = [self.optimizer['D'], self.loss['D']]
        if with_summary and self.discriminator_summary is not None:
            fetches.append(self.discriminator_summary)

        angle_labels = labels
        # angle_labels, radius_labels = tf.split(
        #     labels, [self.number_of_angles, self.radius_label_size])

        feed_dict = {
            self.batch_size: [len(fake_images)],
            self.images: images,
            self.discriminator_labels: discriminator_labels,
            self.angle_labels: angle_labels,
            K.learning_phase(): 1,
        }
        return self.session.run(fetches, feed_dict)[1:]

    def _define_graph(self):
        super(OrbitalGAN, self)._define_graph()

        # To avoid ambguity -- these are the output (e.g. probability) labels
        self.discriminator_labels = self.labels

        batch_size = tf.cast(tf.squeeze(self.batch_size), tf.int32)

        real_latent = self.latent[batch_size:]
        self.angle_labels = Input(batch_shape=[None], dtype=tf.int32)

        real_latent = util.check_numerics(real_latent)

        self.angle_means, self.angle_variances = [], []
        angle_partitions = tf.dynamic_partition(
            real_latent,
            self.angle_labels,
            num_partitions=self.number_of_angles)
        for group in angle_partitions:
            mean, variance = tf.nn.moments(group, axes=[0])
            self.angle_means.append(mean)
            self.angle_variances.append(variance)

        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        update_ema_op = self.ema.apply(self.angle_means + self.angle_variances)

        cosine_distances = []
        for mean, vectors in zip(self.angle_means, angle_partitions):
            moving_mean = tf.expand_dims(self.ema.average(mean), axis=0)
            repetitions = tf.concat([tf.shape(vectors)[:1], [1]], axis=0)
            repeated = tf.tile(moving_mean, tf.cast(repetitions, tf.int32))

            mean_norm = K.l2_normalize(repeated, axis=1)
            vector_norm = K.l2_normalize(vectors, axis=1)
            cosine_similarity = K.sum(mean_norm * vector_norm, axis=1)
            cosine_distance = K.mean(1 - cosine_similarity)
            cosine_distances.append(cosine_distance)

        self.cosine_distance = tf.reduce_mean(cosine_distances)
        self.loss['D'] += 0.01 * self.cosine_distance

        # mean_variance = tf.reduce_mean(
        #     [self.ema.average(v) for v in self.angle_variances])
        #
        # self.loss['D'] += 1 * mean_variance

        # self.radius_labels = Input(shape=[self.radius_label_shape])

        with tf.control_dependencies([update_ema_op]):
            origin_mask = tf.equal(self.angle_labels, self.origin_label)
            origin_vectors = tf.boolean_mask(real_latent, origin_mask)
            self.origin_norm = tf.norm(tf.reduce_mean(origin_vectors, axis=0))

        # self.loss['D'] += 0.01 * self.origin_norm
        # self.loss['O'] = origin_norm

    def train_on_batch(self, batch, with_summary=False):
        real_images, labels = batch
        real_images = (real_images * 2.0) - 1
        batch_size = len(real_images)
        fake_images = self.generate(batch_size, rescale=False)

        d_tensors = self._train_discriminator(fake_images, real_images, labels,
                                              with_summary)
        g_tensors = self._train_generator(batch_size, None, with_summary)

        losses = dict(D=d_tensors[0], G=g_tensors[0])
        return self._maybe_with_summary(losses, g_tensors, d_tensors,
                                        with_summary)

    def _add_summaries(self):
        super(OrbitalGAN, self)._add_summaries()
        with K.name_scope('summary/D'):
            tf.summary.scalar('origin_norm', self.origin_norm)
            tf.summary.scalar('cosine_distance', self.cosine_distance)
            for index, variance in enumerate(self.angle_variances):
                tf.summary.scalar('angle-variance-{0}'.format(index),
                                  tf.norm(self.ema.average(variance)))

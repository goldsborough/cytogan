import tensorflow as tf
import keras.backend as K
import numpy as np

from cytogan.models import model


def _merge_summaries(scope):
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)
    return tf.summary.merge(summaries)


class GAN(model.Model):
    def __init__(self, hyper, learning, session):
        assert len(hyper.image_shape) == 3
        # Copy all fields from hyper to self.
        for index, field in enumerate(hyper._fields):
            setattr(self, field, hyper[index])

        self.image_shape = list(hyper.image_shape)
        self.number_of_channels = hyper.image_shape[-1]
        self.flat_image_shape = np.prod(hyper.image_shape)

        self.images = None  # x
        self.conditional_input = None
        self.batch_size = None
        self.noise = None  # z

        self.generator = None  # G(z, c)
        self.discriminator = None  # D(x)
        self.encoder = None
        self.gan = None  # D(G(z, c))

        super(GAN, self).__init__(learning, session)

        self.generator_summary = _merge_summaries('G')
        self.discriminator_summary = _merge_summaries('D')
        self.summary = tf.summary.merge(
            [self.generator_summary, self.discriminator_summary])

    @property
    def name(self):
        _name = super(GAN, self).name
        if self.conditional_input is None:
            return _name
        else:
            return 'Conditional {0}'.format(_name)

    @property
    def learning_rate(self):
        learning_rates = {}
        for key, lr in self._learning_rate.items():
            if isinstance(lr, tf.Tensor):
                lr = lr.eval(session=self.session)
            learning_rates[key] = lr
        return learning_rates

    def encode(self, images):
        return self.encoder.predict_on_batch(np.array(images))

    def generate(self, latent_samples, conditionals=None, rescale=True):
        feed_dict = {K.learning_phase(): 0}
        if isinstance(latent_samples, int):
            feed_dict[self.batch_size] = [latent_samples]
        else:
            feed_dict[self.noise] = latent_samples
        if conditionals is not None:
            feed_dict[self.conditional_input] = conditionals
        images = self.session.run(self.fake_images, feed_dict)
        # Go from [-1, +1] scale back to [0, 1]
        return (images + 1) / 2 if rescale else images

    def train_on_batch(self, batch, with_summary=False):
        if self.conditional_input is None:
            real_images, conditionals = batch, None
        else:
            real_images, conditionals = batch

        real_images = (real_images * 2) - 1
        batch_size = len(real_images)
        fake_images = self.generate(batch_size, conditionals, rescale=False)
        assert real_images.shape == fake_images.shape, (real_images.shape,
                                                        fake_images.shape)

        d_tensors = self._train_discriminator(fake_images, real_images,
                                              with_summary, conditionals)
        g_tensors = self._train_generator(batch_size, with_summary,
                                          conditionals)

        losses = dict(D=d_tensors[0], G=g_tensors[0])

        if with_summary:
            summary = self._get_combined_summaries(g_tensors[1], d_tensors[1])
            return losses, summary
        else:
            return losses

    def _get_combined_summaries(self, generator_summary,
                                discriminator_summary):
        return self.session.run(
            self.summary,
            feed_dict={
                self.generator_summary: generator_summary,
                self.discriminator_summary: discriminator_summary,
            })

    def _add_optimizer(self, learning):
        self.optimizer = {}
        self._learning_rate = {}
        initial_learning_rate = learning.rate
        if isinstance(initial_learning_rate, float):
            initial_learning_rate = [initial_learning_rate] * 2

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with K.name_scope('D_opt'):
            self._learning_rate['D'] = self._get_learning_rate_tensor(
                initial_learning_rate[0], learning.decay,
                learning.steps_per_decay)
            with tf.control_dependencies(update_ops):
                self.optimizer['D'] = tf.train.AdamOptimizer(
                    self._learning_rate['D'], beta1=0.5).minimize(
                        self.loss['D'],
                        var_list=self.discriminator.trainable_weights)

        with K.name_scope('G_opt'):
            self._learning_rate['G'] = self._get_learning_rate_tensor(
                initial_learning_rate[1], learning.decay,
                learning.steps_per_decay)
            with tf.control_dependencies(update_ops):
                self.optimizer['G'] = tf.train.AdamOptimizer(
                    self._learning_rate['G'], beta1=0.5).minimize(
                        self.loss['G'],
                        var_list=self.generator.trainable_weights,
                        global_step=self.global_step)

    def __repr__(self):
        lines = [self.name]
        try:
            # >= Keras 2.0.6
            self.generator.summary(print_fn=lines.append)
            self.discriminator.summary(print_fn=lines.append)
        except TypeError:
            lines = [layer.name for layer in self.generator.layers]
            lines = [layer.name for layer in self.discriminator.layers]
        return '\n'.join(map(str, lines))

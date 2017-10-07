import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Input

from cytogan.models import model


def _merge_summaries(scope):
    scope = 'summary/{0}'.format(scope)
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)
    return tf.summary.merge(summaries) if summaries else None


def get_conditional_inputs(scopes, conditional_shape):
    inputs = {}
    for scope in scopes:
        name = '{0}/conditional'.format(scope)
        inputs[scope] = Input(shape=conditional_shape, name=name)

    return inputs


def smooth_labels(labels, low=0.8, high=1.0):
    with K.name_scope('noisy_labels'):
        return labels * tf.random_uniform(tf.shape(labels), low, high)


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
        self.conditional = None
        self.batch_size = None
        self.noise = None  # z
        self.fake_images = None  # G(z)

        self.generator = None  # G(z)
        self.discriminator = None  # D(x)
        self.encoder = None
        self.gan = None  # D(G(z, c))

        super(GAN, self).__init__(learning, session)

        self.generator_summary = _merge_summaries('G')
        self.discriminator_summary = _merge_summaries('D')
        if (self.generator_summary is not None
                and self.discriminator_summary is not None):
            self.summary = tf.summary.merge(
                [self.generator_summary, self.discriminator_summary])

    @property
    def name(self):
        _name = super(GAN, self).name
        if self.is_conditional:
            return 'Conditional {0}'.format(_name)
        else:
            return _name

    @property
    def is_conditional(self):
        if not hasattr(self, 'conditional_shape'):
            return False
        return self.conditional_shape is not None

    @property
    def learning_rate(self):
        learning_rates = {}
        for key, lr in self._learning_rate.items():
            if isinstance(lr, tf.Tensor):
                lr = lr.eval(session=self.session)
            learning_rates[key] = lr
        return learning_rates

    def encode(self, batch):
        images, conditionals = self._expand_batch(batch)
        if conditionals is None:
            return self.encoder.predict_on_batch(images)
        return self.encoder.predict_on_batch([images, conditionals])

    def generate(self, latent_samples, conditionals=None, rescale=True):
        feed_dict = {K.learning_phase(): 0}
        if isinstance(latent_samples, int):
            feed_dict[self.batch_size] = [latent_samples]
        else:
            feed_dict[self.noise] = latent_samples
        if conditionals is not None:
            feed_dict[self.conditional['G']] = conditionals
        images = self.session.run(self.fake_images, feed_dict)
        # Go from [-1, +1] scale back to [0, 1]
        return (images + 1) / 2.0 if rescale else images

    def train_on_batch(self, batch, with_summary=False):
        real_images, conditionals = self._expand_batch(batch)
        real_images = (real_images * 2.0) - 1
        batch_size = len(real_images)
        fake_images = self.generate(batch_size, conditionals, rescale=False)

        d_tensors = self._train_discriminator(fake_images, real_images,
                                              conditionals, with_summary)
        g_tensors = self._train_generator(batch_size, conditionals,
                                          with_summary)

        losses = dict(D=d_tensors[0], G=g_tensors[0])
        return self._maybe_with_summary(losses, g_tensors, d_tensors,
                                        with_summary)

    def _get_combined_summary(self, generator_summary, discriminator_summary):
        return self.session.run(
            self.summary,
            feed_dict={
                self.generator_summary: generator_summary,
                self.discriminator_summary: discriminator_summary,
            })

    def _get_model_parameters(self, is_conditional):
        generator_inputs = [self.batch_size]
        discriminator_inputs = [self.images]
        generator_outputs = [self.fake_images]
        if is_conditional:
            generator_inputs.append(self.conditional['G'])
            generator_outputs.append(self.conditional['G'])
            discriminator_inputs.append(self.conditional['D'])

        return generator_inputs, discriminator_inputs, generator_outputs

    def _add_optimizer(self, learning):
        self.optimizer = {}
        self._learning_rate = {}
        initial_learning_rate = learning.rate
        if isinstance(initial_learning_rate, float):
            initial_learning_rate = [initial_learning_rate] * 2

        with K.name_scope('opt/D'):
            self._learning_rate['D'] = self._get_learning_rate_tensor(
                initial_learning_rate[0], learning.decay,
                learning.steps_per_decay)
            self.optimizer['D'] = tf.train.AdamOptimizer(
                self._learning_rate['D'], beta1=0.5).minimize(
                    self.loss['D'],
                    var_list=self.discriminator.trainable_weights)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with K.name_scope('opt/G'):
            self._learning_rate['G'] = self._get_learning_rate_tensor(
                initial_learning_rate[1], learning.decay,
                learning.steps_per_decay)
            with tf.control_dependencies(update_ops):
                self.optimizer['G'] = tf.train.AdamOptimizer(
                    self._learning_rate['G'], beta1=0.5).minimize(
                        self.loss['G'],
                        var_list=self.generator.trainable_weights,
                        global_step=self.global_step)

    def _maybe_with_summary(self, losses, g_tensors, d_tensors, with_summary):
        if with_summary:
            if len(g_tensors) == len(d_tensors) > 1:
                summary = self._get_combined_summary(g_tensors[1],
                                                     d_tensors[1])
            elif len(g_tensors) > 1:
                summary = g_tensors[1]
            elif len(d_tensors) > 1:
                summary = d_tensors[1]
            else:
                summary = None
            return losses, summary
        else:
            return losses

    def _add_summaries(self):
        with K.name_scope('summary/G'):
            tf.summary.histogram('noise', self.noise)
            tf.summary.scalar('loss', self.loss['G'])
            tf.summary.image(
                'generated_images', self.fake_images, max_outputs=4)

        with K.name_scope('summary/D'):
            tf.summary.histogram('latent', self.latent)
            tf.summary.scalar('loss', self.loss['D'])

    def _expand_batch(self, batch):
        if self.is_conditional:
            return np.array(batch[0]), np.array(batch[1])
        else:
            return np.array(batch), None

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

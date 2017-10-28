import tensorflow as tf
import matplotlib.pyplot as plot
import numpy as np
import scipy.stats


class DataDistribution(object):
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def sample(self, N):
        samples = np.random.normal(self.mean, self.sigma, N)
        samples.sort()
        return samples.reshape(-1, 1)


class GeneratorDistribution(object):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def sample(self, N):
        points = np.linspace(self.lower, self.upper, N)
        perturbations = np.random.random(N) * 0.01
        samples = points * perturbations
        return samples.reshape(-1, 1)


def linear(input, hidden_size, name_scope):
    with tf.variable_scope(name_scope):
        weights = tf.get_variable(
            'w',
            shape=[input.shape[1], hidden_size],
            initializer=tf.random_normal_initializer(stddev=0.1))
        bias = tf.get_variable(
            'b', shape=[hidden_size], initializer=tf.constant_initializer(0))
    return tf.matmul(input, weights) + bias


def generator(input, hidden_size):
    h0 = tf.nn.elu(linear(input, hidden_size, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1


def discriminator(input, hidden_size):
    h0 = tf.nn.elu(linear(input, hidden_size * 2, 'd0'))
    h1 = tf.nn.elu(linear(h0, hidden_size * 2, 'd1'))
    h2 = tf.nn.elu(linear(h1, hidden_size * 2, 'd2'))
    h3 = tf.nn.elu(linear(h2, hidden_size * 2, 'd3'))
    h4 = tf.sigmoid(linear(h3, 1, 'd4'))
    return h4


def optimizer(loss, variables):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        0.001,
        decay_rate=0.95,
        decay_steps=1000,
        global_step=global_step,
        staircase=True)
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step, var_list=variables)


def plot_results(session, generator, z, data, gen, n=100):
    x_data = data.sample(n)
    p_data = scipy.stats.norm.pdf(x_data, loc=data.mean, scale=data.sigma)

    z_gen = gen.sample(n)
    x_gen = session.run(generator, feed_dict={z: z_gen})
    bins = np.linspace(gen.lower - 2, gen.upper + 2, n + 1)
    p_gen, _ = np.histogram(x_gen, bins=bins, density=True)

    plot.plot(x_data.flatten(), p_data.flatten(), label='data')
    plot.plot(bins[:-1], p_gen, label='generated')
    plot.legend(loc='upper left')
    plot.show()


batch_size = 128
number_of_epochs = 5000

with tf.variable_scope('G'):
    z = tf.placeholder(tf.float32, shape=(None, 1))
    G = generator(z, 4)

with tf.variable_scope('D') as scope:
    x = tf.placeholder(tf.float32, shape=(None, 1))
    D1 = discriminator(x, 8)
    scope.reuse_variables()
    D2 = discriminator(G, 8)

loss_d = tf.reduce_mean(-tf.log(D1) - tf.log(1 - G))
loss_g = tf.reduce_mean(-tf.log(D2))

vars = tf.trainable_variables()
d_params = [v for v in vars if v.name.startswith('D/')]
g_params = [v for v in vars if v.name.startswith('G/')]

opt_d = optimizer(loss_d, d_params)
opt_g = optimizer(loss_g, g_params)

data = DataDistribution(mean=4, sigma=0.5)
gen = GeneratorDistribution(lower=2, upper=6)

with tf.Session() as session:
    tf.global_variables_initializer().run(session=session)

    for step in range(number_of_epochs):
        for _ in range(1):
            x_sample = data.sample(batch_size)
            z_sample = gen.sample(batch_size)
            d, _ = session.run(
                [loss_d, opt_d], feed_dict={x: x_sample,
                                            z: z_sample})

            if np.isnan(d):
                raise RuntimeError('D is NaN')

        z_sample = gen.sample(batch_size)
        g, _ = session.run([loss_g, opt_g], feed_dict={z: z_sample})

        if np.isnan(d):
            raise RuntimeError('G is NaN')

        print('D: {0} | G: {1}'.format(d, g))

    plot_results(session, G, z, data, gen)

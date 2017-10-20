import tensorflow as tf
import numpy as np


def merge_summaries(scope):
    scope = 'summary/{0}'.format(scope)
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)
    return tf.summary.merge(summaries) if summaries else None


def tf_print(op, tensors):
    def print_message(x):
        print(x)
        return x

    prints = []
    for tensor in tensors:
        prints.append(tf.py_func(print_message, [tensor], tensor.dtype))
    with tf.control_dependencies(prints):
        return tf.identity(op)


def check_numerics(tensor, message=''):
    with tf.control_dependencies([tf.check_numerics(tensor, message)]):
        return tf.identity(tensor)


def check_range(tensor, low, high, message_prefix=''):
    low = tf.assert_greater_equal(tensor, low, message=message_prefix + '>=')
    high = tf.assert_less_equal(tensor, high, message=message_prefix + '<=')
    with tf.control_dependencies([low, high]):
        return tf.identity(tensor)


def top_k(tensor, k):
    top = []
    for _ in range(k):
        argmin = tf.cast(tf.argmin(tensor), tf.int32)
        top.append(tensor[argmin])
        tensor = tf.concat([tensor[:argmin], tensor[argmin + 1:]], axis=-1)
    return top


def binary_labels(zeros, ones):
    return np.concatenate([np.zeros(zeros), np.ones(ones)], axis=0)

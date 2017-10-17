import tensorflow as tf

def merge_summaries(scope):
    scope = 'summary/{0}'.format(scope)
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)
    return tf.summary.merge(summaries) if summaries else None

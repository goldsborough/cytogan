import tensorflow as tf

x = tf.Variable(0)

saver = tf.train.Saver()
with tf.Session() as session:
    tf.global_variables_initializer().run(session=session)
    x = tf.assign_add(x, 5)
    saver.save(session, 'chkp')

with tf.Session() as session:
    saver.restore(session, 'chkp')
    tf.global_variables_initializer().run(session=session)
    print(x.eval(session=session))

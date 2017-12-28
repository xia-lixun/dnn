import tensorflow as tf
import numpy as np

# a = (b + c) * (c + 2)

const = tf.constant(2.0, name='const')
# b = tf.Variable(2.0, name='b')
b = tf.placeholder(tf.float32, [None, 1], name='b')
c = tf.Variable(1.0, name='c')

d = tf.add(b,c,name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d,e,name='a')

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    a_out = sess.run(a, feed_dict={b: np.arange(0,10)[:,np.newaxis]})
    print("Variable is {}".format(a_out)) # shall be 9.0
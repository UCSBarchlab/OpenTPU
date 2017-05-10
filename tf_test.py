import tensorflow as tf
import numpy as np

inputs = tf.placeholder(np.float32, [2, 2])

m = tf.Variable(np.ones((2, 2))*32, dtype=tf.float32)

op = tf.matmul(inputs, m)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

input_val = [[1, 1], [2, 1]]

print 'inputs: {}\nmatrix: {}'.format(input_val, sess.run(m))
print sess.run(op, feed_dict={inputs: input_val})

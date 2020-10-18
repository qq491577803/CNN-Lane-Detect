import tensorflow as tf
import numpy as np
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])


data = {'x':np.array([1]),'y':np.array([2])}
def plus(a,b):
        c = a+b
        return c
with tf.Session() as sess:
    plus = plus(x, y)
    for i in range(5):
        result = sess.run(plus,feed_dict={x:data['x'],y:data['y']})

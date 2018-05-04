#! /usr/bin/env python

import tensorflow as tf
import itertools

tf.set_random_seed(0)

data, targ = [tf.random_uniform([1, 1]) for _ in range(2)]
loss = tf.square(tf.layers.dense(data, 1) - targ)
train = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in itertools.count():
    print(i)
    print(sess.run([loss, train]))

from sac.networks import network_interface
import tensorflow as tf
import numpy as np


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha*x)

class MLPValueFunc(object):

    def Q_network(self, s, a, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            sa = tf.concat([s, a], axis=1)
            fc1 = tf.layers.dense(sa, 128, tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 128, tf.nn.relu, name='fc2')
            q = tf.reshape(tf.layers.dense(fc2, 1, name='q'), [-1])
        return q

    def V_network(self, s, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            fc1 = tf.layers.dense(s, 128, tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 128, tf.nn.relu, name='fc2')
            v = tf.reshape(tf.layers.dense(fc2, 1, name='v'), [-1])
        return v

class CNN_Goal_ValueFunc(object):

    def Q_network(self, s, a, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            a_shape = a.get_shape()[1]
            c1 = tf.layers.conv2d(s, 32, 5, activation=tf.nn.relu, strides=2, padding='SAME', name='c1')  # 14 x 14 x 32
            c2 = tf.layers.conv2d(c1, 32, 5, activation=tf.nn.relu, strides=2, padding='SAME', name='c2')  # 7 x 7 x 32
            flat = tf.reshape(c2, [-1, 7*7*32])
            enc = tf.layers.dense(flat, 128, activation=tf.nn.relu, name='fc1')
            all_q = tf.layers.dense(enc, a_shape, name='all_q')
            q = tf.reduce_sum(a * all_q, axis=1)
        return q

    def V_network(self, s, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            c1 = tf.layers.conv2d(s, 32, 5, activation=tf.nn.relu, strides=2, padding='SAME', name='c1')  # 14 x 14 x 32
            c2 = tf.layers.conv2d(c1, 32, 5, activation=tf.nn.relu, strides=2, padding='SAME', name='c2')  # 7 x 7 x 32
            flat = tf.reshape(c2, [-1, 7*7*32])
            enc = tf.layers.dense(flat, 128, activation=tf.nn.relu, name='fc1')
            v = tf.reshape(tf.layers.dense(enc, 1, name='v'), [-1])
        return v




from sac.networks import network_interface
import tensorflow as tf
import numpy as np


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)


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

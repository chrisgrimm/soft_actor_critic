from sac.networks import network_interface
import tensorflow as tf
import numpy as np


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)


class MLPValueFunc(object):
    def Q_network(self, s, a, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            sa = tf.concat([s, a], axis=1)
            fc1 = tf.layers.dense(sa, 256, tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 256, tf.nn.relu, name='fc2')
            fc3 = tf.layers.dense(fc2, 256, tf.nn.relu, name='fc3')
            q = tf.reshape(tf.layers.dense(fc3, 1, name='q'), [-1])
        return q

    def V_network(self, s, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            fc1 = tf.layers.dense(s, 256, tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 256, tf.nn.relu, name='fc2')
            fc3 = tf.layers.dense(fc2, 256, tf.nn.relu, name='fc3')
            v = tf.reshape(tf.layers.dense(fc3, 1, name='v'), [-1])
        return v

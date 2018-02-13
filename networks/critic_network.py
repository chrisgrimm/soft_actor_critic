import tensorflow as tf
import numpy as np


class CriticValueNetwork(object):

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size


    def Q(self, state1, action, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            fc0 = tf.concat([state1, action], axis=1)
            fc1 = tf.layers.dense(fc0, self.hidden_size, tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, self.hidden_size, tf.nn.relu, name='fc2')
            Q = tf.layers.dense(fc2, 1, name='Q')
        return Q

    def V(self, state1, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            fc1 = tf.layers.dense(state1, self.hidden_size, tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, self.hidden_size, tf.nn.relu, name='fc2')
            V = tf.layers.dense(fc2, 1, name='Q')
        return V

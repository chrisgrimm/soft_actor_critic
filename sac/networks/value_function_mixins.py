from networks import network_interface
import tensorflow as tf
import numpy as np
from networks.utils import power2_encoding


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha*x)

def MLPValueFunc(width, depth):
    class MLPValueFunc(object):

        def Q_network(self, s, a, name, reuse=None):
            with tf.variable_scope(name, reuse=reuse):
                sa = tf.concat([s, a], axis=1)
                fc_i = sa
                for i in range(depth):
                    fc_i = tf.layers.dense(fc_i, width, tf.nn.relu, name=f'fc{i+1}')
                #fc1 = tf.layers.dense(sa, width, tf.nn.relu, name='fc1')
                #fc2 = tf.layers.dense(fc1, width, tf.nn.relu, name='fc2')
                q = tf.reshape(tf.layers.dense(fc_i, 1, name='q'), [-1])
            return q

        def V_network(self, s, name, reuse=None):
            with tf.variable_scope(name, reuse=reuse):
                fc_i = s
                for i in range(depth):
                    fc_i = tf.layers.dense(fc_i, width, tf.nn.relu, name=f'fc{i+1}')
                #fc1 = tf.layers.dense(s, width, tf.nn.relu, name='fc1')
                #fc2 = tf.layers.dense(fc1, width, tf.nn.relu, name='fc2')
                v = tf.reshape(tf.layers.dense(fc_i, 1, name='v'), [-1])
            return v
    return MLPValueFunc

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

class CNN_Power2_ValueFunc(object):

    def Q_network(self, s, a, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            a_shape = a.get_shape()[1].value
            enc = power2_encoding(s)
            a_fc1 = tf.layers.dense(a, 128, activation=tf.nn.relu, name='a_fc1')
            combined = tf.concat([enc, a_fc1], axis=1)
            q = tf.reshape(tf.layers.dense(combined, 1, name='q'), [-1])
        return q

    def V_network(self, s, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            enc = power2_encoding(s)
            v = tf.reshape(tf.layers.dense(enc, 1, name='v'), [-1])
        return v


'''class MLP_Categorical_X_Gaussian_ValueFunc(object):

    def Q_network(self, s, a, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            print(a.get_shape()[1].value)
            assert a.get_shape()[1].value == 9
            #a_cat, a_gauss = a[:, :8], a[:, 8:]
            #a = tf.concat([a_cat, a_gauss], axis=1)
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
        return v'''



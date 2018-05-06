import tensorflow as tf

from sac.utils import ACT


class MLPValueFunc(object):
    def Q_network(self, s, a, name='Q', reuse=None):
        activation = tf.nn.relu
        with tf.variable_scope(name, reuse=reuse):
            sa = tf.concat([s, a], axis=1)
            fc1 = tf.layers.dense(sa, 256, activation, name='fc1')
            print(fc1)
            fc2 = tf.layers.dense(fc1, 256, activation, name='fc2')
            print(fc2)
            fc3 = tf.layers.dense(fc2, 256, activation, name='fc3')
            print(fc3)
            # fc4 = tf.layers.dense(fc3, 256, activation, name='fc4')
            # fc5 = tf.layers.dense(fc4, 256, activation, name='fc5')
            dense = tf.layers.dense(fc3, 1, name='q')
            print(dense)
            reshape = tf.reshape(dense, [-1])
            print(reshape)
            q = reshape
        return q

    def V_network(self, s, name='V', reuse=None):
        activation = tf.nn.relu
        with tf.variable_scope(name, reuse=reuse):
            fc1 = tf.layers.dense(s, 256, activation, name='fc1')
            print(fc1)
            fc2 = tf.layers.dense(fc1, 256, activation, name='fc2')
            print(fc2)
            fc3 = tf.layers.dense(fc2, 256, activation, name='fc3')
            print(fc3)
            # fc4 = tf.layers.dense(fc3, 256, activation, name='fc4')
            # fc5 = tf.layers.dense(fc4, 256, activation, name='fc5')
            dense = tf.layers.dense(fc3, 1, name='v')
            print(dense)
            reshape = tf.reshape(dense, [-1])
            print(reshape)
            v = reshape
        return v

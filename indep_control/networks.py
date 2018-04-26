from indep_control import network_interface
from indep_control.utils import component

import tensorflow as tf

class MINetwork(network_interface.AbstractIndependenceNetwork):

    @component
    def independence_criterion(self, Z1_joint, Z2_joint, Z1_marg, Z2_marg):
        T_joint, _ = self.T_network(Z1_joint, Z2_joint, name='T')
        T_marg, _ = self.T_network(Z1_marg, Z2_marg, name='T', reuse=True)
        MI = tf.reduce_mean(T_joint) - tf.log(tf.reduce_mean(tf.exp(T_marg)))
        return tf.maximum(MI, 0), -MI

    @component
    def T_network(self, Z1, Z2):
        #(enc_Z1, _), (enc_Z2, _) = self._encoder(Z1, 10, name='enc'), self._encoder(Z2, 10, name='enc', reuse=True)
        enc_Z1, enc_Z2 = 10*Z1, 10*Z2
        Z = tf.concat([enc_Z1, enc_Z2], axis=1)
        fc1 = tf.layers.dense(Z, 500, tf.nn.leaky_relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 500, tf.nn.leaky_relu, name='fc2')
        T = tf.reshape(tf.layers.dense(fc2, 1, name='T'), [-1])
        return T

    @component
    def _encoder(self, inp, bottleneck):
        #bn = lambda inp: tf.layers.batch_normalization(inp, training=self.inp_training)
        bn = lambda x: x
        print('inp', inp)
        c1 = bn(tf.layers.conv2d(inp, 32, 5, 2, 'SAME', activation=tf.nn.leaky_relu, name='c1'))  # 14 x 14 x 32
        c2 = bn(tf.layers.conv2d(c1, 32, 5, 2, 'SAME', activation=tf.nn.leaky_relu, name='c2'))  # 7 x 7 x 32
        enc = tf.layers.dense(tf.reshape(c2, [-1, 7 * 7 * 32]), bottleneck, activation=tf.nn.leaky_relu, name='enc')
        return enc


class GANNetwork(network_interface.AbstractIndependenceNetwork):

    @component
    def independence_criterion(self, Z1_joint, Z2_joint, Z1_marg, Z2_marg):
        DX, _ = self.D(Z1_joint, None, name='D')
        DGZ, _ = self.D(Z1_marg, None, name='D', reuse=True)
        loss = tf.reduce_mean(tf.log(DX + 0.00001) + tf.log(1 - DGZ + 0.00001))
        #loss = -0.5*tf.reduce_mean(tf.square(DX - 1) + tf.square(DGZ))
        return loss, -loss
    #moop
    @component
    def D(self, Z1, Z2):
        (enc_Z1, _) = self._encoder(Z1, 10, name='enc')#self._encoder(Z2, 10, name='enc', reuse=True)
        #Z = tf.concat([enc_Z1, enc_Z2], axis=1)
        #Z = tf.concat([Z1, Z2], axis=1)
        fc1 = tf.layers.dense(enc_Z1, 500, tf.nn.leaky_relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 500, tf.nn.leaky_relu, name='fc2')
        D = tf.reshape(tf.layers.dense(fc2, 1, tf.nn.sigmoid, name='D'), [-1])
        return D

    @component
    def _encoder(self, inp, bottleneck):
        # bn = lambda inp: tf.layers.batch_normalization(inp, training=self.inp_training)
        bn = lambda x: x
        print('inp', inp)
        c1 = bn(tf.layers.conv2d(inp, 32, 5, 2, 'SAME', activation=tf.nn.leaky_relu, name='c1'))  # 14 x 14 x 32
        c2 = bn(tf.layers.conv2d(c1, 32, 5, 2, 'SAME', activation=tf.nn.leaky_relu, name='c2'))  # 7 x 7 x 32
        enc = tf.layers.dense(tf.reshape(c2, [-1, 7 * 7 * 32]), bottleneck, activation=tf.nn.leaky_relu, name='enc')
        return enc
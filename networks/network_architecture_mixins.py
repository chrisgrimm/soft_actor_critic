import tensorflow as tf

EPS = 1E-6


class MLPPolicy(object):

    def input_processing(self, s):
        fc1 = tf.layers.dense(s, 128, tf.nn.relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 128, tf.nn.relu, name='fc2')
        return fc2


class CNNPolicy(object):

    def input_processing(self, s):
        # assumes s is 32 x 32 x 3
        c1 = tf.layers.conv2d(s, 32, 5, 2, 'SAME', activation=tf.nn.relu, name='c1')  # 16 x 16 x 32
        c2 = tf.layers.conv2d(c1, 32, 5, 2, 'SAME', activation=tf.nn.relu, name='c2')  # 8 x 8 x 32
        c3 = tf.layers.conv2d(c2, 32, 5, 2, 'SAME', activation=tf.nn.relu, name='c3')  # 4 x 4 x 32
        return tf.reshape(c3, [-1, 4 * 4 * 32])



def MLPPolicyGen(num_layers, neurons, activation):
    class Arch(object):
        def input_processing(self, s):
            x = s
            for i in range(num_layers):
                x = tf.layers.dense(s, neurons, activation, name='fc'+str(i))
            return x
    return Arch




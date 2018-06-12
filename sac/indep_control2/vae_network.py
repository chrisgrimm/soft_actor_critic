import tensorflow as tf
import numpy as np
from utils import component, horz_stack_images

class VAE_Network(object):

    def __init__(self, hidden_size, input_size, mode):
        assert mode in ['vector', 'image']
        self.hidden_size = hidden_size
        self.input_size = input_size
        beta = 1000

        image_size = 128
        if mode == 'vector':
            self.inp = inp = tf.placeholder(tf.float32, [None, self.input_size], 'inp')
            self.inp_flat = inp_flat = self.inp
        else:
            self.inp = inp = tf.placeholder(tf.float32, [None, image_size, image_size, 3], name='inp')
            self.inp_flat = inp_flat = tf.reshape(self.inp, [-1, image_size*image_size*3])

        self.inp_C = inp_C = tf.placeholder(tf.float32)

        self.is_training = tf.placeholder(tf.bool)
        bs = tf.shape(self.inp)[0]
        if mode == 'vector':
            (enc_mean, enc_log_std_sq), enc_vars = self.encoder(self.inp, name='encoder')
        else:
            (enc_mean, enc_log_std_sq), enc_vars = self.encoder_conv(self.inp, name='encoder')

        #N = tf.distributions.Normal(tf.zeros([bs, hidden_size]), tf.ones([bs, hidden_size])).sample()
        N = tf.random_normal([bs, hidden_size])
        self.Z_sample = Z_sample = (tf.sqrt(tf.exp(enc_log_std_sq)) * N) + enc_mean
        self.Z_mean = enc_mean

        if mode == 'vector':
            dec_mean, dec_vars = self.decoder(Z_sample, name='decoder')
            dec_mean_flat = dec_mean
        else:
            dec_mean, dec_vars = self.decoder_conv(Z_sample, name='decoder')
            dec_mean_flat = tf.reshape(dec_mean, [-1, image_size*image_size*3])
        self.X_sample = dec_mean

        #term1 = tf.reduce_mean(tf.reduce_sum(dec_dist.log_prob(self.inp), axis=1) + tf.reduce_sum(prior_dist.log_prob(Z_sample), axis=1), axis=0)
        KL = -0.5 * tf.reduce_sum(1 + enc_log_std_sq
                                           - tf.square(enc_mean)
                                           - tf.exp(enc_log_std_sq), 1)
        term1 = -beta * tf.abs(KL - inp_C)
        term2 = tf.reduce_sum(inp_flat * tf.log(1e-7 + dec_mean_flat) + (1-inp_flat) * tf.log(1e-7 + 1 - dec_mean_flat), 1)

        #term1 = 0.5*tf.reduce_sum(1 + tf.log(tf.square(tf.exp(enc_log_std))) - tf.square(enc_mean) - tf.square(tf.exp(enc_log_std)), axis=1)
        #term2 = tf.reduce_sum(dec_dist.log_prob(self.inp), axis=1)
        #term2 = tf.reduce_mean(tf.reduce_sum(enc_dist.log_prob(Z_sample), axis=1), axis=0)
        self.loss = loss = tf.reduce_mean(term1 + term2)

        # maximize variational bound.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = train_op = tf.train.AdamOptimizer(learning_rate=5e-4).minimize(-loss)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        self.saver = tf.train.Saver()
        self.sess = sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

    def encode(self, X):
        [Z_sample] = self.sess.run([self.Z_sample], feed_dict={self.inp: X, self.is_training: False})
        return Z_sample

    def encode_deterministic(self, X):
        [Z_mean] = self.sess.run([self.Z_mean], feed_dict={self.inp: X, self.is_training: False})
        return Z_mean

    def decode(self, Z):
        [X_sample] = self.sess.run([self.X_sample], feed_dict={self.Z_sample: Z, self.is_training: False})
        return X_sample

    def autoencode(self, X):
        [X_sample] = self.sess.run([self.X_sample], feed_dict={self.inp: X, self.is_training: False})
        return X_sample

    def train(self, X, C):
        [loss, _] = self.sess.run([self.loss, self.train_op], feed_dict={self.inp: X, self.is_training: True, self.inp_C: C})
        return loss

    def save(self, file_path):
        self.saver.save(self.sess, file_path)

    def restore(self, file_path):
        self.saver.restore(self.sess, file_path)


    @component
    def encoder(self, inp):
        #bn = lambda x: tf.layers.batch_normalization(x, training=self.is_training)
        bn = lambda x: x
        fc1 = tf.nn.leaky_relu(bn(tf.layers.dense(inp, 500, name='fc1')))
        fc2 = tf.nn.leaky_relu(bn(tf.layers.dense(fc1, 500, name='fc2')))
        enc_mean = tf.layers.dense(fc2, self.hidden_size, name='enc_mean')
        enc_log_sigma_sq = tf.layers.dense(fc2, self.hidden_size, name='enc_cov')
        return enc_mean, enc_log_sigma_sq

    @component
    def encoder_conv(self, inp):
        #bn = lambda x: tf.layers.batch_normalization(x, training=self.is_training)
        bn = lambda x: x
        c = tf.nn.relu(bn(tf.layers.conv2d(inp, 32, 4, 2, 'SAME', name='c'))) # 64 x 64 x 32
        c0 = tf.nn.relu(bn(tf.layers.conv2d(c, 32, 4, 2, 'SAME', name='c0'))) # 32 x 32 x 32
        c1 = tf.nn.relu(bn(tf.layers.conv2d(c0, 32, 4, 2, 'SAME', name='c1'))) # 16 x 16 x 32
        c2 = tf.nn.relu(bn(tf.layers.conv2d(c1, 32, 4, 2, 'SAME', name='c2'))) # 8 x 8 x 32
        size = np.product([x.value for x in c2.get_shape()[1:]])
        c2_flat = tf.reshape(c2, [-1, size])
        fc1 = tf.nn.relu(bn(tf.layers.dense(c2_flat, 256, name='fc1')))
        fc2 = tf.nn.relu(bn(tf.layers.dense(fc1, 256, name='fc2')))
        enc_mean = tf.layers.dense(fc2, self.hidden_size, name='enc_mean')
        enc_log_sigma_sq = tf.layers.dense(fc2, self.hidden_size, name='enc_cov')
        return enc_mean, enc_log_sigma_sq

    @component
    def decoder_conv(self, enc):
        #bn = lambda x: tf.layers.batch_normalization(x, training=self.is_training)
        bn = lambda x: x
        fc1 = tf.nn.relu(bn(tf.layers.dense(enc, 256, name='fc1')))
        fc2 = tf.nn.relu(bn(tf.layers.dense(fc1, 8*8*32, name='fc2')))
        d1 = tf.nn.relu(bn(tf.layers.conv2d_transpose(tf.reshape(fc2, [-1, 8, 8, 32]), 32, 4, 2, 'SAME', name='d1'))) # 16 x 16 x 32
        d2 = tf.nn.relu(bn(tf.layers.conv2d_transpose(d1, 32, 4, 2, 'SAME', name='d2'))) # 32 x 32 x 32
        d3 = tf.nn.relu(bn(tf.layers.conv2d_transpose(d2, 32, 4, 2, 'SAME', name='d3'))) # 64 x 64 x 32

        d4 = tf.layers.conv2d_transpose(d3, 3, 4, 2, 'SAME', activation=tf.nn.sigmoid, name='d4') # 128 x 128 x 3
        return d4

    @component
    def decoder(self, inp):
        #bn = lambda x: tf.layers.batch_normalization(x, training=self.is_training)
        bn = lambda x: x
        fc1 = tf.nn.leaky_relu(bn(tf.layers.dense(inp, 500, name='fc1')))
        fc2 = tf.nn.leaky_relu(bn(tf.layers.dense(fc1, 500, name='fc2')))
        dec_mean = tf.layers.dense(fc2, self.input_size, activation=tf.nn.sigmoid, name='dec_mean')
        return dec_mean



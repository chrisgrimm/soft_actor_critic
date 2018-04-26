import tensorflow as tf
from indep_control.utils import component
import numpy as np
from abc import abstractmethod

class AbstractIndependenceNetwork(object):

    def __init__(self):
        learning_rate = 0.0001
        bottleneck_size = 20
        indep_weighting = 1.0
        self.inp_image1 = inp_image1 = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='inp_image1')
        self.inp_image2 = inp_image2 = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='inp_image2')
        self.inp_image3 = inp_image3 = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='inp_image3')
        self.inp_training = tf.placeholder(tf.bool)

        bs = tf.shape(self.inp_image1)[0]
        # encode both input images
        encoded1, encoder_vars = self.encoder(inp_image1, bottleneck_size, name='enc')
        self.encoded1 = encoded1
        encoded2, _ = self.encoder(inp_image2, bottleneck_size, name='enc', reuse=True)
        encoded3, _ = self.encoder(inp_image3, bottleneck_size, name='enc', reuse=True)

        # decode first image.

        [enc1, enc2] = self.feature_splitter(encoded1, 2)
        pre_random = tf.concat([tf.expand_dims(enc1, -1), tf.expand_dims(enc2, -1)], axis=2)
        idx = tf.random_uniform(shape=[bs], minval=0, maxval=2, dtype=tf.int32)
        first_onehot = tf.reshape(tf.one_hot(idx, 2), [-1, 1, 2])  # [bs, 1, 2]        R_enc1, R_enc2 = shuffled[0], shuffled[1]
        second_onehot = tf.reshape(tf.one_hot(1 - idx, 2), [-1, 1, 2])
        choice1 = tf.reduce_sum(first_onehot * pre_random, axis=2)
        choice2 = tf.reduce_sum(second_onehot * pre_random, axis=2)
        R_encoded1 = tf.concat([choice1, choice2], axis=1)

        decoded, decoder_vars = self.decoder(encoded1, name='dec')
        self.decoded = decoded
        # collect autoencoder variables
        ae_vars = encoder_vars + decoder_vars



        # split the encoded vectors.
        [enc1_Z1, enc1_Z2] = self.feature_splitter(encoded1, 2)
        [enc2_Z1, enc2_Z2] = self.feature_splitter(encoded2, 2)
        [C1, C2] = self.feature_splitter(encoded3, 2)

        pre_random = tf.concat([tf.expand_dims(enc1_Z1, -1), tf.expand_dims(enc1_Z2, -1)], axis=2)
        idx_1 = tf.random_uniform(shape=[bs], minval=0, maxval=2, dtype=tf.int32)
        idx_2 = tf.random_uniform(shape=[bs], minval=0, maxval=2, dtype=tf.int32)

        onehot_idx_1 = tf.reshape(tf.one_hot(idx_1, 2), [-1, 1, 2]) # [bs, 1, 2]
        onehot_idx_2 = tf.reshape(tf.one_hot(idx_2, 2), [-1, 1, 2]) # [bs, 1, 2]



        R_enc1_Z1 = tf.reduce_sum(onehot_idx_1 * pre_random, axis=2)
        R_enc1_Z2 = tf.reduce_sum(onehot_idx_2 * pre_random, axis=2)



        pre_random = tf.concat([tf.expand_dims(enc2_Z1, -1), tf.expand_dims(enc2_Z2, -1)], axis=2)
        idx_1 = tf.random_uniform(shape=[bs], minval=0, maxval=2, dtype=tf.int32)
        idx_2 = tf.random_uniform(shape=[bs], minval=0, maxval=2, dtype=tf.int32)

        onehot_idx_1 = tf.reshape(tf.one_hot(idx_1, 2), [-1, 1, 2])  # [bs, 1, 2]
        onehot_idx_2 = tf.reshape(tf.one_hot(idx_2, 2), [-1, 1, 2])  # [bs, 1, 2]


        R_enc2_Z1 = tf.reduce_sum(onehot_idx_1 * pre_random, axis=2)
        R_enc2_Z2 = tf.reduce_sum(onehot_idx_2 * pre_random, axis=2)

        self.combined_marg = tf.concat([enc1_Z1, enc2_Z2], axis=1)


        # Naming scheme: X(Z_i^j, C_k) = X_Zij_Ck

        #decoded1_joint, _ = self.decoder(tf.concat([C1, enc1_Z2], axis=1), name='dec', reuse=True)
        #decoded2_joint, _ = self.decoder(tf.concat([enc1_Z1, C2], axis=1), name='dec', reuse=True)
        #decoded1_marg, _ = self.decoder(tf.concat([C1, enc2_Z2], axis=1), name='dec', reuse=True)
        #decoded2_marg, _ = self.decoder(tf.concat([enc1_Z1, C2], axis=1), name='dec', reuse=True)

        decoded_joint, _ = self.decoder(tf.concat([enc1_Z1, enc1_Z2], axis=1), name='dec', reuse=True)
        decoded_marg, _ = self.decoder(tf.concat([enc1_Z1, enc2_Z2], axis=1), name='dec', reuse=True)

        #decoded_dep1, _ = self.decoder(tf.concat([enc1_Z1, tf.tile(tf.reduce_mean(enc1_Z2, axis=0, keep_dims=True), [bs, 1])], axis=1), name='dec', reuse=True)
        #decoded_dep2, _ = self.decoder(tf.concat([tf.tile(tf.reduce_mean(enc1_Z1, axis=0, keep_dims=True), [bs, 1]), enc1_Z2], axis=1), name='dec', reuse=True)


        #X_Z11_C2, _ = self.decoder(tf.concat([enc1_Z1, C2], axis=1), name='dec', reuse=True)
        #X_C1_Z21, _ = self.decoder(tf.concat([C1, enc1_Z2], axis=1), name='dec', reuse=True)
        #, _ = self.decoder(tf.concat([enc1_Z1, C2], axis=1), name='dec', reuse=True)
        #X_C1_Z22, _ = self.decoder(tf.concat([C1, enc2_Z2], axis=1), name='dec', reuse=True)

        (indep_term_ae, indep_term), indep_vars = self.independence_criterion(
            decoded_joint, None, decoded_marg, None, name='indep')

        #(dep1_term_ae, dep1_term), dep1_vars = self.independence_criterion(
        #    decoded_joint, None, decoded_dep1, None, name='dep1')

        #(dep2_term_ae, dep2_term), dep2_vars = self.independence_criterion(
        #    decoded_joint, None, decoded_dep2, None, name='dep2')


        print(tf.gradients(decoded_joint, enc1_Z1))
        self.G1 = tf.reduce_mean(tf.norm(tf.gradients(decoded_joint, enc1_Z1), axis=1))
        self.G2 = tf.reduce_mean(tf.norm(tf.gradients(decoded_joint, enc1_Z2), axis=1))


        #X_Z11_Z22, _ = self.decoder(tf.concat([enc1_Z1, enc2_Z2], axis=1), name='dec', reuse=True)
        #X_Z12_Z21, _ = self.decoder(tf.concat([enc2_Z1, enc1_Z2], axis=1), name='dec', reuse=True)
        #X_Z12_C2, _ = self.decoder(tf.concat([enc2_Z1, C2], axis=1), name='dec', reuse=True)

#        (indep_term_ae, indep_term), indep_vars = self.independence_criterion(
#            X_Z11_C2, X_C1_Z21, X_Z11_C2, X_C1_Z22, name='indep')

        #(dep1_term_ae, dep1_term), dep1_vars = self.independence_criterion(
        #    X_Z11_C2, X_Z11_Z22, X_Z11_C2, X_C1_Z22, name='dep1')

        #(dep2_term_ae, dep2_term), dep2_vars = self.independence_criterion(
        #    X_C1_Z21, X_Z12_Z21, X_C1_Z21, X_Z12_C2, name='dep2')

        self.recombined, _ = self.decoder(tf.concat([enc1_Z1, enc2_Z2], axis=1), name='dec', reuse=True)



        Z1_joint, Z2_joint = enc1_Z1, enc1_Z2
        Z1_marg, Z2_marg = enc1_Z1, enc2_Z2

        # create independence criteria for autoencoder and indep losses.
        self.ae_loss = ae_loss = tf.reduce_mean(tf.square(decoded - inp_image1)) + indep_weighting * (indep_term_ae)# - dep1_term_ae - dep2_term_ae)#- tf.minimum(dep1_term_ae, 1) - tf.minimum(dep2_term_ae, 1))
        self.indep_loss = indep_loss = tf.reduce_mean(indep_term)# + dep1_term + dep2_term)# + dep1_term + dep2_term)


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_ae = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(ae_loss, var_list=ae_vars)
            self.train_indep = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(indep_loss, var_list=indep_vars)# + dep1_vars + dep2_vars)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        self.saver = tf.train.Saver()
        self.sess = sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())


    def train(self, X, Y, Z):
        [_, _, autoencoder_loss, indep_loss, G1, G2] = self.sess.run([self.train_ae, self.train_indep, self.ae_loss, self.indep_loss, self.G1, self.G2], feed_dict={self.inp_image1: X, self.inp_image2: Y, self.inp_image3: Z, self.inp_training: True})
        print('gradients', G1, G2)
        return autoencoder_loss, -indep_loss

    def autoencode(self, X):
        [out] = self.sess.run([self.decoded], feed_dict={self.inp_image1: X, self.inp_training: False})
        return out

    def encode_single(self, X):
        [encoding] = self.sess.run([self.encoded1], feed_dict={self.inp_image1: X})
        return encoding

    def encode_together(self, X1, X2):
        [encoding] = self.sess.run([self.combined_marg], feed_dict={self.inp_image1: X1, self.inp_image2: X2})
        return encoding


    def recombine(self, X, Y):
        [decoded] = self.sess.run([self.recombined], feed_dict={self.inp_image1: X, self.inp_image2: Y, self.inp_training: False})
        return decoded


    def feature_splitter(self, enc, num_splits):
        enc_shape = enc.get_shape()[1].value
        assert enc_shape % num_splits == 0
        split_size = enc_shape // num_splits
        split_features = []
        for i in range(num_splits):
            enc_split = enc[:, i*split_size:(i+1)*split_size]
            split_features.append(enc_split)
        return split_features

    def interpolate_noise(self, z, other_z, other_z_pred):
        bs = tf.shape(z)[0]
        z_size = tf.shape(z)[1]
        noise = tf.random_uniform([bs, z_size], -1, 1)
        # when we can predict the other z, how_predictive is going to be 0.
        temperature = 1.
        sq = tf.square(other_z - other_z_pred)
        sq = tf.Print(sq, [sq], 'square')
        how_predictive = 1 - 2*(tf.sigmoid(temperature*sq) - 0.5)
        #how_predictive =
        how_predictive = tf.Print(how_predictive, [how_predictive], 'how_predictive')
        return z*(1 - how_predictive) + noise*(how_predictive)


    def save(self, file_path):
        self.saver.save(self.sess, file_path)

    def restore(self, file_path):
        self.saver.restore(self.sess, file_path)


    @abstractmethod
    def independence_criterion(self, Z1_joint, Z2_joint, Z1_marg, Z2_marg):
        pass


    @component
    def encoder(self, inp, bottleneck):
        #bn = lambda inp: tf.layers.batch_normalization(inp, training=self.inp_training)
        bn = lambda x: x
        c1 = bn(tf.layers.conv2d(inp, 32, 5, 2, 'SAME', activation=tf.nn.tanh, name='c1')) # 14 x 14 x 32
        c2 = bn(tf.layers.conv2d(c1, 32, 5, 2, 'SAME', activation=tf.nn.tanh, name='c2')) # 7 x 7 x 32
        enc = tf.layers.dense(tf.reshape(c2, [-1, 7*7*32]), bottleneck, activation=tf.nn.sigmoid, name='enc')
        return enc

    @component
    def decoder(self, inp):
        #bn = lambda inp: tf.layers.batch_normalization(inp, training=self.inp_training)
        bn = lambda x: x
        post_enc = tf.reshape(bn(tf.layers.dense(inp, 7*7*32, tf.nn.tanh, name='post_enc')), [-1, 7, 7, 32])
        dc1 = bn(tf.layers.conv2d_transpose(post_enc, 32, 5, 2, 'SAME', activation=tf.nn.tanh, name='dc1'))
        out = tf.layers.conv2d_transpose(dc1, 3, 5, 2, 'SAME', activation=tf.nn.sigmoid, name='out')
        return out

    @component
    def feature_predictor(self, feature1, feature2):
        #bn = lambda inp: tf.layers.batch_normalization(inp, training=self.inp_training)
        bn = lambda x: x
        feature2_size = feature2.get_shape()[1].value
        fc1 = bn(tf.layers.dense(feature1, 100, tf.nn.leaky_relu, name='fc1'))
        fc2 = bn(tf.layers.dense(fc1, 100, tf.nn.leaky_relu, name='fc2'))

        pred_feature2 = tf.layers.dense(fc2, feature2_size, tf.nn.tanh, name='pred_feature2')
        pred_loss = tf.reduce_mean(tf.square(pred_feature2 - feature2))

        return pred_feature2, pred_loss







import tensorflow as tf
import numpy as np

class ActorNetwork(object):

    def __init__(self, hidden_size, num_gaussians):
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians


    def log_pi(self, state1, action, name, reuse=None):
        self.mus, self.log_sigmas, self.ws = [], [], []
        with tf.variable_scope(name, reuse=reuse):
            for i in range(self.num_gaussians):
                mu, log_sigma, w = self.build_gaussian_head(state1, 'gaussian_%s' % (i+1))
                self.mus.append(mu)
                self.log_sigmas.append(log_sigma)
                self.ws.append(w)
        return self.tf_log_prob(action)


    def tf_log_prob(self, action):
        normalizing_factor = tf.reduce_sum(self.ws, axis=0) # [None, 1]
        terms = []
        for mu, log_sigma, w in zip(self.mus, self.log_sigmas, self.ws):
            head_pdf = tf.distributions.Normal(mu, tf.exp(log_sigma)).prob(action)
            term = w * head_pdf
            terms.append(term)
        pi_unbounded = tf.reduce_sum(terms, axis=0) / normalizing_factor
        pi_bounded = tf.log(pi_unbounded) - tf.reduce_sum(tf.log(1.0 - tf.tanh(action)), axis=1)
        return pi_bounded


    def build_gaussian_head(self, state1, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):

            fc1_mu = tf.layers.dense(state1, self.hidden_size, activation=tf.nn.relu, name='fc1_mu')
            fc2_mu = tf.layers.dense(fc1_mu, self.hidden_size, activation=tf.nn.relu, name='fc2_mu')
            mu = tf.layers.dense(fc2_mu, self.action_shape, name='mu')

            fc1_sigma = tf.layers.dense(state1, self.hidden_size, activation=tf.nn.relu, name='fc1_sigma')
            fc2_sigma = tf.layers.dense(fc1_sigma, self.hidden_size, activation=tf.nn.relu, name='fc2_sigma')
            log_sigma = tf.layers.dense(fc2_sigma, self.action_shape, name='log_sigma')

            fc1_w = tf.layers.dense(state1, self.hidden_size, activation=tf.nn.relu, name='fc1_w')
            fc2_w = tf.layers.dense(fc1_w, self.hidden_size, activation=tf.nn.relu, name='fc2_w')
            w = tf.layers.dense(fc2_w, 1, activation=tf.nn.relu, name='w')

            return mu, log_sigma, w
















import tensorflow as tf
import numpy as np
class ActorNetwork(object):

    def __init__(self, action_shape, hidden_size, num_gaussians, is_continuous=True):
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians
        self.action_shape = action_shape
        self.is_continuous = is_continuous


    def log_pi_gaussian(self, state1, action, name, reuse=None):
        self.mus, self.sigmas, self.ws = [], [], []
        with tf.variable_scope(name, reuse=reuse):
            for i in range(self.num_gaussians):
                mu, sigma, w = self.build_gaussian_head(state1, 'gaussian_%s' % (i+1))
                self.mus.append(mu)
                self.sigmas.append(sigma)
                self.ws.append(w)
                print(w)
        return self.tf_log_prob(action)

    def log_pi_categorical_discrete_ethan(self, state1, action, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            fc1 = tf.layers.dense(state1, self.hidden_size, activation=tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, self.hidden_size, activation=tf.nn.relu, name='fc2')
            logits = tf.layers.dense(fc2, self.action_shape[0], name='logits')
            self.probs = tf.nn.softmax(logits)
            log_prob = tf.reduce_sum(tf.log(self.probs) * action, axis=1)
        return log_prob

    def sample_discrete(self):
        return tf.distributions.Categorical(probs=self.probs).sample()


    def log_pi(self, state1, action, name, reuse=None):
        if self.is_continuous:
            return self.log_pi_gaussian(state1, action, name, reuse)
        else:
            return self.log_pi_categorical_discrete_ethan(state1, action, name, reuse)


    def tf_log_prob(self, action):
        normalizing_factor = tf.reduce_sum(self.ws, axis=0) # [None, 1]
        terms = []
        for mu, sigma, w in zip(self.mus, self.sigmas, self.ws):
            head_pdf = tf.distributions.Normal(mu, sigma).prob(action)
            print('head_pdf', head_pdf)
            term = w * head_pdf
            terms.append(term)
        print('terms', terms[0])
        pi_unbounded = tf.reduce_sum(terms, axis=0) / normalizing_factor
        print(pi_unbounded)

        pi_bounded = tf.log(pi_unbounded) - tf.reduce_sum(tf.log(1.0 - tf.square(tf.tanh(action))), axis=1)
        return pi_bounded




    def build_gaussian_head(self, state1, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):

            fc1_mu = tf.layers.dense(state1, self.hidden_size, activation=tf.nn.relu, name='fc1_mu')
            fc2_mu = tf.layers.dense(fc1_mu, self.hidden_size, activation=tf.nn.relu, name='fc2_mu')
            mu = tf.layers.dense(fc2_mu, self.action_shape[0], name='mu')

            fc1_sigma = tf.layers.dense(state1, self.hidden_size, activation=tf.nn.relu, name='fc1_sigma')
            fc2_sigma = tf.layers.dense(fc1_sigma, self.hidden_size, activation=tf.nn.relu, name='fc2_sigma')
            sigma = tf.layers.dense(fc2_sigma, self.action_shape[0], activation=tf.abs, name='log_sigma')

            fc1_w = tf.layers.dense(state1, self.hidden_size, activation=tf.nn.relu, name='fc1_w')
            fc2_w = tf.layers.dense(fc1_w, self.hidden_size, activation=tf.nn.relu, name='fc2_w')
            w = tf.layers.dense(fc2_w, 1, activation=tf.abs, name='w')

            return mu, sigma, w
















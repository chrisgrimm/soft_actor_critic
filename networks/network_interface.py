import tensorflow as tf
import numpy as np
from abc import abstractmethod


class AbstractSoftActorCritic(object):

    def __init__(self, s_shape, a_shape):
        self.S1 = S1 = tf.placeholder(tf.float32, [None] + list(s_shape))
        self.S2 = S2 = tf.placeholder(tf.float32, [None] + list(s_shape))
        self.A = A = tf.placeholder(tf.float32, [None] + list(a_shape))
        self.R = R = tf.placeholder(tf.float32, [None])
        self.T = T = tf.placeholder(tf.float32, [None])
        gamma = 0.99
        tau = 0.01
        learning_rate = 3*10**-4

        # constructing V loss

        self.A_sampled = A_sampled = tf.stop_gradient(self.sample_pi_network(a_shape[0], S1, 'pi'))
        V_S1 = self.V_network(S1, 'V')
        Q_sampled = self.Q_network(S1, A_sampled, 'Q')
        log_pi_sampled = self.pi_network_log_prob(A_sampled, S1, 'pi', reuse=True)
        self.V_loss = V_loss = tf.reduce_mean(0.5*tf.square(V_S1 - (Q_sampled - log_pi_sampled)))

        # constructing Q loss
        V_bar_S2 = self.V_network(S2, 'V_bar')
        Q = self.Q_network(S1, A, 'Q', reuse=True)
        self.Q_loss = Q_loss = tf.reduce_mean(0.5*tf.square(Q - (R + gamma * V_bar_S2)))

        # constructing pi loss
        self.pi_loss = pi_loss = tf.reduce_mean(log_pi_sampled * tf.stop_gradient(log_pi_sampled - Q_sampled + V_S1))

        # grabbing all the relevant variables
        phi = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi/')
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q/')
        xi = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='V/')
        xi_bar = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='V_bar/')

        print('\nphi', phi)
        print('\ntheta', theta)
        print('\nxi', xi)
        print('\nxi_bar', xi_bar)

        soft_update_xi_bar_ops = [tf.assign(xbar, tau*x + (1 - tau)*xbar) for (xbar, x) in zip(xi_bar, xi)]
        self.soft_update_xi_bar = soft_update_xi_bar = tf.group(*soft_update_xi_bar_ops)
        hard_update_xi_bar_ops = [tf.assign(xbar, x) for (xbar, x) in zip(xi_bar, xi)]
        hard_update_xi_bar = tf.group(*hard_update_xi_bar_ops)

        self.train_V = train_V = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(V_loss, var_list=xi)
        self.train_Q = train_Q = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Q_loss, var_list=theta)
        self.train_pi = train_pi = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(pi_loss, var_list=phi)
        self.check = tf.add_check_numerics_ops()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        # ensure that xi and xi_bar are the same at initialization
        sess.run(hard_update_xi_bar)

    def train_step(self, S1, A, R, S2, T):
        [_, _, _, _, V_loss, Q_loss, pi_loss] = self.sess.run(
            [self.check, self.train_Q, self.train_V, self.train_pi, self.V_loss, self.Q_loss, self.pi_loss],
            feed_dict={self.S1: S1, self.A: A, self.R: R, self.S2: S2, self.T: T})
        self.sess.run(self.soft_update_xi_bar)
        return V_loss, Q_loss, pi_loss

    def sample_actions(self, S1):
        actions = self.sess.run(self.A_sampled, feed_dict={self.S1: S1})
        return actions

    @abstractmethod
    def Q_network(self, s, a, name, reuse=None):
        pass

    @abstractmethod
    def V_network(self, s, name, reuse=None):
        pass

    @abstractmethod
    def input_processing(self, s):
        pass

    @abstractmethod
    def produce_policy_parameters(self, a_shape, processed_s):
        pass

    @abstractmethod
    def policy_parameters_to_log_prob(self, a, parameters):
        pass

    @abstractmethod
    def policy_parameters_to_sample(self, parameters):
        pass

    def pi_network_log_prob(self, a, s, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            processed_s = self.input_processing(s)
            a_shape = a.get_shape()[1].value
            parameters = self.produce_policy_parameters(a_shape, processed_s)
            log_prob = self.policy_parameters_to_log_prob(a, parameters)
        return log_prob

    def sample_pi_network(self, a_shape, s, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            processed_s = self.input_processing(s)
            parameters = self.produce_policy_parameters(a_shape, processed_s)
            sample = self.policy_parameters_to_sample(parameters)
        return sample
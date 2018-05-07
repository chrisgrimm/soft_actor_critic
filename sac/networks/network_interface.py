import tensorflow as tf
from abc import abstractmethod


def mlp(inputs, layer_size, n_layers, activation):
    for i in range(n_layers):
        inputs = tf.layers.dense(
            inputs, layer_size, activation, name='fc' + str(i))
    return inputs


class AbstractSoftActorCritic(object):
    def __init__(self, s_shape, a_shape, activation: str, n_layers: int,
                 layer_size: int, learning_rate: float):
        self.activation = dict(
            relu=tf.nn.relu,
            crelu=tf.nn.crelu,
            selu=tf.nn.selu,
            elu=tf.nn.elu,
            leaky=tf.nn.leaky_relu,
            leaky_relu=tf.nn.leaky_relu,
            tanh=tf.nn.tanh,
        )[activation]
        self.n_layers = n_layers
        self.layer_size = layer_size

        tf.set_random_seed(0)
        self.S1 = S1 = tf.placeholder(
            tf.float32, [None] + list(s_shape), name='S1')
        self.S2 = S2 = tf.placeholder(
            tf.float32, [None] + list(s_shape), name='S2')
        self.A = A = tf.placeholder(
            tf.float32, [None] + list(a_shape), name='A')
        self.R = R = tf.placeholder(tf.float32, [None], name='R')
        self.T = T = tf.placeholder(tf.float32, [None], name='T')
        gamma = 0.99
        tau = 0.01
        # learning_rate = 3 * 10 ** -4

        self.A_max_likelihood = tf.stop_gradient(
            self.get_best_action(a_shape[0], S1, 'pi'))

        # constructing V loss
        with tf.control_dependencies([self.A_max_likelihood]):
            self.A_sampled1 = A_sampled1 = tf.stop_gradient(
                self.sample_pi_network(a_shape[0], S1, 'pi', reuse=True))
            V_S1 = self.V_network(S1, 'V')
            Q_sampled1 = self.Q_network(
                S1, self.transform_action_sample(A_sampled1), 'Q')
            log_pi_sampled1 = self.pi_network_log_prob(
                A_sampled1, S1, 'pi', reuse=True)
            self.entropy = self.entropy_from_sa(A_sampled1, S1)
            self.V_loss = V_loss = tf.reduce_mean(
                0.5 * tf.square(V_S1 - (Q_sampled1 - log_pi_sampled1)))

        # constructing Q loss
        with tf.control_dependencies([self.V_loss]):
            V_bar_S2 = self.V_network(S2, 'V_bar')
            Q = self.Q_network(
                S1, self.transform_action_sample(A), 'Q', reuse=True)
            self.Q_loss = Q_loss = tf.reduce_mean(
                0.5 * tf.square(Q - (R + (1 - T) * gamma * V_bar_S2)))

        # constructing pi loss
        with tf.control_dependencies([self.Q_loss]):
            self.A_sampled2 = A_sampled2 = tf.stop_gradient(
                self.sample_pi_network(a_shape[0], S1, 'pi', reuse=True))
            Q_sampled2 = self.Q_network(
                S1, self.transform_action_sample(A_sampled2), 'Q', reuse=True)
            log_pi_sampled2 = self.pi_network_log_prob(
                A_sampled2, S1, 'pi', reuse=True)
            self.pi_loss = pi_loss = tf.reduce_mean(
                log_pi_sampled2 *
                tf.stop_gradient(log_pi_sampled2 - Q_sampled2 + V_S1))

        # grabbing all the relevant variables
        phi = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi/')
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q/')
        xi = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='V/')
        xi_bar = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='V_bar/')

        with tf.control_dependencies([self.pi_loss]):
            self.train_V = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(
                    V_loss, var_list=xi)
        with tf.control_dependencies([self.train_V]):
            self.train_Q = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(
                    Q_loss, var_list=theta)
        with tf.control_dependencies([self.train_Q]):
            self.train_pi = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(
                    pi_loss, var_list=phi)

        with tf.control_dependencies([self.train_pi]):
            soft_update_xi_bar_ops = [
                tf.assign(xbar, tau * x + (1 - tau) * xbar)
                for (xbar, x) in zip(xi_bar, xi)
            ]
            self.soft_update_xi_bar = tf.group(*soft_update_xi_bar_ops)
            self.check = tf.add_check_numerics_ops()
            # ensure that xi and xi_bar are the same at initialization

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        # ensure that xi and xi_bar are the same at initialization
        hard_update_xi_bar_ops = [
            tf.assign(xbar, x) for (xbar, x) in zip(xi_bar, xi)
        ]

        hard_update_xi_bar = tf.group(*hard_update_xi_bar_ops)
        sess.run(hard_update_xi_bar)

    def train_step(self, S1, A, R, S2, T):
        [entropy, _, _, _, _, V_loss, Q_loss, pi_loss] = self.sess.run(
            [
                self.entropy, self.soft_update_xi_bar, self.train_V,
                self.train_Q, self.train_pi, self.V_loss, self.Q_loss,
                self.pi_loss
            ],
            feed_dict={
                self.S1: S1,
                self.A: A,
                self.R: R,
                self.S2: S2,
                self.T: T
            })
        return entropy, V_loss, Q_loss, pi_loss

    def get_actions(self, S1, sample=True):
        if sample:
            actions = self.sess.run(self.A_sampled1, feed_dict={self.S1: S1})
        else:
            actions = self.sess.run(
                self.A_max_likelihood, feed_dict={self.S1: S1})
        return actions[0]

    def mlp(self, inputs):
        return mlp(
            inputs=inputs,
            layer_size=self.layer_size,
            n_layers=self.n_layers,
            activation=self.activation)

    def Q_network(self, s, a, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            sa = tf.concat([s, a], axis=1)
            return tf.reshape(tf.layers.dense(self.mlp(sa), 1, name='q'), [-1])

    def V_network(self, s, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            return tf.reshape(tf.layers.dense(self.mlp(s), 1, name='v'), [-1])

    def input_processing(self, s):
        return self.mlp(s)

    @abstractmethod
    def produce_policy_parameters(self, a_shape, processed_s):
        pass

    @abstractmethod
    def policy_parameters_to_log_prob(self, a, parameters):
        pass

    @abstractmethod
    def policy_parameters_to_sample(self, parameters):
        pass

    @abstractmethod
    def policy_parameters_to_max_likelihood_action(self, parameters):
        pass

    @abstractmethod
    def transform_action_sample(self, action_sample):
        pass

    @abstractmethod
    def entropy_from_params(self, params):
        pass

    def entropy_from_sa(self, a, s, reuse=None):
        with tf.variable_scope('entropy', reuse=reuse):
            processed_s = self.input_processing(s)
            a_shape = a.get_shape()[1].value
            parameters = self.produce_policy_parameters(a_shape, processed_s)
        return tf.reduce_mean(self.entropy_from_params(parameters))


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

    def get_best_action(self, a_shape, s, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            processed_s = self.input_processing(s)
            parameters = self.produce_policy_parameters(a_shape, processed_s)
            actions = self.policy_parameters_to_max_likelihood_action(
                parameters)
        return actions

from typing import Tuple, Iterable, Callable

from abc import abstractmethod

import tensorflow as tf
from collections import namedtuple

from sac.utils import PropStep, Step
import numpy as np


def mlp(inputs, layer_size, n_layers, activation):
    for i in range(n_layers):
        inputs = tf.layers.dense(
            inputs, layer_size, activation, name='fc' + str(i))
    return inputs


TRAIN_VALUES = 'entropy soft_update_xi_bar train_V train_Q' \
               ' train_pi V_loss Q_loss pi_loss V_grad Q_grad pi_grad'.split()
TrainStep = namedtuple('TrainStep', TRAIN_VALUES)


class AbstractAgent:
    def __init__(self, s_shape: Iterable, a_shape: Iterable, activation: Callable, n_layers: int,
                 layer_size: int, learning_rate: float) -> None:
        self.activation = activation
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

        with tf.variable_scope('pi'):
            processed_s = self.input_processing(S1)
            self.parameters = self.produce_policy_parameters(
                a_shape[0], processed_s)

        # generate actions:
        self.A_max_likelihood = tf.stop_gradient(self.get_best_action('pi'))
        self.A_sampled1 = A_sampled1 = tf.stop_gradient(
            self.sample_pi_network('pi', reuse=True))

        # constructing V loss
        with tf.control_dependencies([self.A_sampled1]):
            V_S1 = self.V_network(S1, 'V')
            Q_sampled1 = self.Q_network(
                S1, self.transform_action_sample(A_sampled1), 'Q')
            log_pi_sampled1 = self.pi_network_log_prob(
                A_sampled1, 'pi', reuse=True)
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
                self.sample_pi_network('pi', reuse=True))
            Q_sampled2 = self.Q_network(
                S1, self.transform_action_sample(A_sampled2), 'Q', reuse=True)
            log_pi_sampled2 = self.pi_network_log_prob(
                A_sampled2, 'pi', reuse=True)
            self.pi_loss = pi_loss = tf.reduce_mean(
                log_pi_sampled2 *
                tf.stop_gradient(log_pi_sampled2 - Q_sampled2 + V_S1))

        # grabbing all the relevant variables
        phi = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi/')
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q/')
        xi = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='V/')
        xi_bar = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='V_bar/')

        def train_op(var_list, loss, dependency):
            with tf.control_dependencies([dependency]):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
                grads = [tf.reduce_mean(g) for g, v in grads_and_vars]
                op = optimizer.apply_gradients(grads_and_vars)
                return grads, op

        self.V_grad, self.train_V = train_op(var_list=xi, loss=V_loss, dependency=self.pi_loss)
        self.Q_grad, self.train_Q = train_op(var_list=theta, loss=Q_loss, dependency=self.train_V)
        self.pi_grad, self.train_pi = train_op(var_list=phi, loss=pi_loss, dependency=self.train_Q)

        with tf.control_dependencies([self.train_pi]):
            soft_update_xi_bar_ops = [
                tf.assign(xbar, tau * x + (1 - tau) * xbar)
                for (xbar, x) in zip(xi_bar, xi)
                ]
            self.soft_update_xi_bar = tf.group(*soft_update_xi_bar_ops)
            self.check = tf.add_check_numerics_ops()
            self.entropy = self.compute_entropy()
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

    def train_step(self, step: Step, extra_feeds: object = None) -> TrainStep:
        feed_dict = {
            self.S1: step.s1,
            self.A: step.a,
            self.R: step.r,
            self.S2: step.s2,
            self.T: step.t
        }
        if extra_feeds:
            feed_dict.update(extra_feeds)
        return TrainStep(*self.sess.run(
            [getattr(self, attr) for attr in TRAIN_VALUES], feed_dict))

    def get_actions(self, S1: np.ndarray, sample: bool = True) -> np.ndarray:
        if sample:
            actions = self.sess.run(self.A_sampled1, feed_dict={self.S1: S1})
        else:
            actions = self.sess.run(
                self.A_max_likelihood, feed_dict={self.S1: S1})
        return actions[0]

    def mlp(self, inputs: tf.Tensor) -> tf.Tensor:
        return mlp(
            inputs=inputs,
            layer_size=self.layer_size,
            n_layers=self.n_layers,
            activation=self.activation)

    def Q_network(self, s: tf.Tensor, a: tf.Tensor, name: str, reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            sa = tf.concat([s, a], axis=1)
            return tf.reshape(tf.layers.dense(self.mlp(sa), 1, name='q'), [-1])

    def V_network(self, s: tf.Tensor, name: str, reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            return tf.reshape(tf.layers.dense(self.mlp(s), 1, name='v'), [-1])

    def V_S1(self) -> tf.Tensor:
        return self.V_network(self.S1, 'V')

    def V_S2(self) -> tf.Tensor:
        return self.V_network(self.S2, 'V_bar')

    def input_processing(self, s: tf.Tensor) -> tf.Tensor:
        return self.mlp(s)

    @abstractmethod
    def produce_policy_parameters(self, a_shape: Iterable, processed_s: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def policy_parameters_to_log_prob(self, a: tf.Tensor, parameters: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def policy_parameters_to_sample(self, parameters: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def policy_parameters_to_max_likelihood_action(self, parameters) -> tf.Tensor:
        pass

    @abstractmethod
    def transform_action_sample(self, action_sample: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def entropy_from_params(self, params: tf.Tensor) -> tf.Tensor:
        pass

    def compute_entropy(self, reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope('entropy', reuse=reuse):
            return tf.reduce_mean(self.entropy_from_params(self.parameters))

    def pi_network_log_prob(self, a: tf.Tensor, name: str, reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            return self.policy_parameters_to_log_prob(a, self.parameters)

    def sample_pi_network(self, name: str, reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            return self.policy_parameters_to_sample(self.parameters)

    def get_best_action(self, name: str, reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            return self.policy_parameters_to_max_likelihood_action(
                self.parameters)


class PropagationAgent(AbstractAgent):
    def __init__(self, **kwargs):
        self.sampled_V2 = tf.placeholder(tf.float32, [None], name='R')
        super().__init__(**kwargs)

    def V_bar_S2(self) -> tf.Tensor:
        return tf.maximum(self.sampled_V2, super().V_S2())

    def train_step(self, step: PropStep, extra_feeds: dict = None) -> Tuple[float, float, float]:
        extra_feeds[self.sampled_V2] = step.v2
        assert isinstance(step, PropStep)
        return super().train_step(step, extra_feeds)

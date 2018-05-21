from typing import Tuple, Iterable, Callable, Sequence

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


TRAIN_VALUES = """\
entropy
soft_update_xi_bar
V_loss
Q_loss
pi_loss
V_grad
Q_grad
pi_grad\
""".split('\n')
TrainStep = namedtuple('TrainStep', TRAIN_VALUES)


class AbstractAgent:
    def __init__(self, s_shape: Iterable, a_shape: Sequence, activation: Callable, n_layers: int,
                 layer_size: int, learning_rate: float, grad_clip: float) -> None:
        self.activation = activation
        self.n_layers = n_layers
        self.layer_size = layer_size

        tf.set_random_seed(0)  # TODO: this needs to go
        self.S1 = tf.placeholder(
            tf.float32, [None] + list(s_shape), name='S1')
        self.S2 = tf.placeholder(
            tf.float32, [None] + list(s_shape), name='S2')
        self.A = A = tf.placeholder(
            tf.float32, [None] + list(a_shape), name='A')
        self.R = R = tf.placeholder(tf.float32, [None], name='R')
        self.T = T = tf.placeholder(tf.float32, [None], name='T')
        gamma = 0.99
        tau = 0.01
        # learning_rate = 3 * 10 ** -4

        with tf.variable_scope('pi'):
            processed_s = self.input_processing(self.S1)
            self.parameters = self.produce_policy_parameters(
                a_shape[0], processed_s)

        # generate actions:
        self.A_max_likelihood = tf.stop_gradient(self.get_best_action('pi'))
        self.A_sampled1 = A_sampled1 = tf.stop_gradient(
            self.sample_pi_network('pi', reuse=True))

        # constructing V loss
        with tf.control_dependencies([self.A_sampled1]):
            v1 = self.compute_v1()
            q1 = self.q_network(
                self.S1, self.transform_action_sample(A_sampled1), 'Q')
            log_pi_sampled1 = self.pi_network_log_prob(
                A_sampled1, 'pi', reuse=True)
            self.V_loss = V_loss = tf.reduce_mean(
                0.5 * tf.square(v1 - (q1 - log_pi_sampled1)))

        # constructing Q loss
        with tf.control_dependencies([self.V_loss]):
            v2 = self.compute_v2()
            q = self.q_network(
                self.S1, self.transform_action_sample(A), 'Q', reuse=True)
            # noinspection PyTypeChecker
            self.Q_loss = Q_loss = tf.reduce_mean(
                0.5 * tf.square(q - (R + (1 - T) * gamma * v2)))

        # constructing pi loss
        with tf.control_dependencies([self.Q_loss]):
            self.A_sampled2 = A_sampled2 = tf.stop_gradient(
                self.sample_pi_network('pi', reuse=True))
            q2 = self.q_network(
                self.S1, self.transform_action_sample(A_sampled2), 'Q', reuse=True)
            log_pi_sampled2 = self.pi_network_log_prob(
                A_sampled2, 'pi', reuse=True)
            self.pi_loss = pi_loss = tf.reduce_mean(
                log_pi_sampled2 *
                tf.stop_gradient(log_pi_sampled2 - q2 + v1))

        # grabbing all the relevant variables
        phi = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi/')
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q/')
        xi = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='V/')
        xi_bar = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='V_bar/')

        def train_op(loss, var_list, dependency):
            with tf.control_dependencies([dependency]):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                gradients, variables = zip(*optimizer.compute_gradients(loss, var_list=var_list))
                if grad_clip:
                    gradients, norm = tf.clip_by_global_norm(gradients, grad_clip)
                else:
                    norm = tf.global_norm(gradients)
                op = optimizer.apply_gradients(zip(gradients, variables))
                return op, norm

        self.train_V, self.V_grad = train_op(loss=V_loss, var_list=xi, dependency=self.pi_loss)
        self.train_Q, self.Q_grad = train_op(loss=Q_loss, var_list=theta, dependency=self.train_V)
        self.train_pi, self.pi_grad = train_op(loss=pi_loss, var_list=phi, dependency=self.train_Q)

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

    def train_step(self, step: Step, feed_dict: dict = None) -> TrainStep:
        if feed_dict is None:
            feed_dict = {
                self.S1: step.s1,
                self.A: step.a,
                self.R: step.r,
                self.S2: step.s2,
                self.T: step.t
            }
        return TrainStep(*self.sess.run(
            [getattr(self, attr) for attr in TRAIN_VALUES], feed_dict))

    def get_actions(self, s1: np.ndarray, sample: bool = True) -> np.ndarray:
        if sample:
            actions = self.sess.run(self.A_sampled1, feed_dict={self.S1: s1})
        else:
            actions = self.sess.run(
                self.A_max_likelihood, feed_dict={self.S1: s1})
        return actions[0]

    def mlp(self, inputs: tf.Tensor) -> tf.Tensor:
        return mlp(
            inputs=inputs,
            layer_size=self.layer_size,
            n_layers=self.n_layers,
            activation=self.activation)

    def q_network(self, s: tf.Tensor, a: tf.Tensor, name: str, reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            sa = tf.concat([s, a], axis=1)
            return tf.reshape(tf.layers.dense(self.mlp(sa), 1, name='q'), [-1])

    def v_network(self, s: tf.Tensor, name: str, reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            return tf.reshape(tf.layers.dense(self.mlp(s), 1, name='v'), [-1])

    def compute_v1(self) -> tf.Tensor:
        return self.v_network(self.S1, 'V')

    def compute_v2(self) -> tf.Tensor:
        return self.v_network(self.S2, 'V_bar')

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


# noinspection PyAbstractClass
class PropagationAgent(AbstractAgent):
    def __init__(self, **kwargs):
        self.sampled_V2 = tf.placeholder(tf.float32, [None], name='R')
        super().__init__(**kwargs)

    def compute_v2(self) -> tf.Tensor:
        return tf.maximum(self.sampled_V2, super().compute_v2())

    def train_step(self, step: PropStep, feed_dict: dict = None) -> TrainStep:
        assert isinstance(step, PropStep)
        if feed_dict is None:
            feed_dict = {
                self.S1: step.s1,
                self.A: step.a,
                self.R: step.r,
                self.S2: step.s2,
                self.T: step.t,
                self.sampled_V2: step.v2,
            }
        return super().train_step(step, feed_dict)

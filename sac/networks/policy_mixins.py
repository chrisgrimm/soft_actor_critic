import tensorflow as tf
import numpy as np
from abc import abstractmethod

EPS = 1E-6


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)


class MLPPolicy(object):
    def input_processing(self, s):
        fc1 = tf.layers.dense(s, 256, tf.nn.relu, name='fc1', kernel_initializer=tf.constant_initializer(np.random.uniform()))
        fc2 = tf.layers.dense(fc1, 256, tf.nn.relu, name='fc2', kernel_initializer=tf.constant_initializer(np.random.uniform()))
        fc3 = tf.layers.dense(fc2, 256, tf.nn.relu, name='fc3', kernel_initializer=tf.constant_initializer(np.random.uniform()))
        return fc3


class GaussianPolicy(object):
    '''
    Policy outputs a gaussian action that is clamped to the interval [-1, 1]
    '''

    def produce_policy_parameters(self, a_shape, processed_s):
        mu_params = tf.layers.dense(processed_s, a_shape, name='mu_params', kernel_initializer=tf.constant_initializer(np.random.uniform()))
        sigma_params = tf.layers.dense(
            processed_s, a_shape, tf.nn.sigmoid, name='sigma_params', kernel_initializer=tf.constant_initializer(np.random.uniform()))
        return (mu_params, sigma_params + 0.0001)

    def policy_parameters_to_log_prob(self, u, parameters):
        (mu, sigma) = parameters
        log_prob = tf.distributions.Normal(mu, sigma).log_prob(u)
        # print(log_prob)
        return tf.reduce_sum(
            log_prob, axis=1) - tf.reduce_sum(
                tf.log(1 - tf.square(tf.tanh(u)) + EPS), axis=1)

    def policy_parameters_to_max_likelihood_action(self, parameters):
        (mu, sigma) = parameters
        return mu

    def policy_parameters_to_sample(self, parameters):
        (mu, sigma) = parameters
        return tf.distributions.Normal(mu, sigma).sample()

    def transform_action_sample(self, action_sample):
        return tf.tanh(action_sample)


class GaussianMixturePolicy(object):
    def produce_policy_parameters(self, a_shape, processed_s):
        pass

    def policy_parmeters_to_log_prob(self, a, parameters):
        pass

    def policy_parameters_to_sample(self, parameters):
        pass


class CategoricalPolicy(object):
    def produce_policy_parameters(self, a_shape, processed_s):
        logits = tf.layers.dense(processed_s, a_shape, name='logits', kernel_initializer=tf.constant_initializer(np.random.uniform()))
        return logits

    def policy_parameters_to_log_prob(self, a, parameters):
        logits = parameters
        out = tf.distributions.Categorical(logits=logits).log_prob(
            tf.argmax(a, axis=1))
        #out = tf.Print(out, [out], summarize=10)
        return out

    def policy_parameters_to_sample(self, parameters):
        logits = parameters
        a_shape = logits.get_shape()[1].value
        #logits = tf.Print(logits, [tf.nn.softmax(logits)], message='logits are:', summarize=10)
        out = tf.one_hot(
            tf.distributions.Categorical(logits=logits).sample(), a_shape)
        return out

    def policy_parameters_to_max_likelihood_action(self, parameters):
        logits = parameters
        a_shape = logits.get_shape()[1].value
        return tf.one_hot(tf.argmax(logits, axis=1), a_shape)

    def transform_action_sample(self, action_sample):
        return action_sample

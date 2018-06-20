import tensorflow as tf
import numpy as np
from abc import abstractmethod
from networks.utils import power2_encoding

EPS = 1E-6

def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha*x)

class MLPPolicy(object):

    def input_processing(self, s):
        fc1 = tf.layers.dense(s, 256, tf.nn.relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 256, tf.nn.relu, name='fc2')
        return fc2

class CNN_Goal_Policy(object):

    def input_processing(self, s):
        c1 = tf.layers.conv2d(s, 32, 5, activation=tf.nn.relu, strides=2, padding='SAME', name='c1') # 14 x 14 x 32
        c2 = tf.layers.conv2d(c1, 32, 5, activation=tf.nn.relu, strides=2, padding='SAME', name='c2') # 7 x 7 x 32
        flat = tf.reshape(c2, [-1, 7*7*32])
        enc = tf.layers.dense(flat, 128, activation=tf.nn.relu, name='fc1')
        return enc


class CNN_Power2_Policy(object):

    def input_processing(self, s):
        return power2_encoding(s)




class GaussianPolicy(object):
    '''
    Policy outputs a gaussian action that is clamped to the interval [-1, 1]
    '''
    def produce_policy_parameters(self, a_shape, processed_s):
        mu_params = tf.layers.dense(processed_s, a_shape, name='mu_params')
        sigma_params = 3*tf.layers.dense(processed_s, a_shape, tf.nn.sigmoid, name='sigma_params')
        return (mu_params, sigma_params + EPS)

    def policy_parameters_to_log_prob(self, u, parameters):
        (mu, sigma) = parameters
        log_prob = tf.distributions.Normal(mu, sigma).log_prob(u)
        #print(log_prob)
        return tf.reduce_sum(log_prob, axis=1) - tf.reduce_sum(tf.log(1 - tf.square(tf.tanh(u)) + EPS), axis=1)

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
        logits = tf.layers.dense(processed_s, a_shape, name='logits')
        return logits

    def policy_parameters_to_log_prob(self, a, parameters):
        logits = parameters
        out = tf.distributions.Categorical(logits=logits).log_prob(tf.argmax(a, axis=1))
        #out = tf.Print(out, [out], summarize=10)
        return out

    def policy_parameters_to_sample(self, parameters):
        logits = parameters
        a_shape = logits.get_shape()[1].value
        #logits = tf.Print(logits, [tf.nn.softmax(logits)], message='logits are:', summarize=10)
        out = tf.one_hot(tf.distributions.Categorical(logits=logits).sample(), a_shape)
        return out

    def policy_parameters_to_max_likelihood_action(self, parameters):
        logits = parameters
        a_shape = logits.get_shape()[1].value
        return tf.one_hot(tf.argmax(logits, axis=1), a_shape)

    def transform_action_sample(self, action_sample):
        return action_sample


class Categorical_X_GaussianPolicy(object):

    def produce_policy_parameters(self, a_shape, processed_s):
        logits_shape = 8
        gaussian_shape = 1
        assert a_shape == 9
        logits = tf.layers.dense(processed_s, logits_shape, name='logits')
        mu_params = tf.layers.dense(processed_s, gaussian_shape, name='mu_params')
        sigma_params = 3 * tf.layers.dense(processed_s, gaussian_shape, tf.nn.sigmoid, name='sigma_params') + EPS
        return (logits, mu_params, sigma_params)

    def policy_parameters_to_log_prob(self, a, parameters):
        # TODO rework this abstraction, so I dont have to hardcode this.
        a_cat = a[:, :8]
        a_gauss = a[:, 8:]
        (logits, mu_params, sigma_params) = parameters
        out_cat = tf.distributions.Categorical(logits=logits).log_prob(tf.argmax(a_cat, axis=1))
        out_gauss = tf.distributions.Normal(mu_params, sigma_params).log_prob(a_gauss)
        out_gauss = tf.reduce_sum(out_gauss, axis=1) - tf.reduce_sum(tf.log(1 - tf.square(tf.tanh(a_gauss)) + EPS), axis=1)
        return out_cat + out_gauss

    def policy_parameters_to_sample(self, parameters):
        (logits, mu_params, sigma_params) = parameters
        logits_shape = logits.get_shape()[1].value
        gaussian_shape = mu_params.get_shape()[1].value
        print('---')
        print(logits)
        print(mu_params)
        print(sigma_params)
        #logits = tf.Print(logits, [tf.nn.softmax(logits)], message='logits are:', summarize=10)
        logits_out = tf.one_hot(tf.distributions.Categorical(logits=logits).sample(), logits_shape)
        gauss_out = tf.distributions.Normal(mu_params, sigma_params).sample()
        a = tf.concat([logits_out, gauss_out], axis=1)
        return a


    def policy_parameters_to_max_likelihood_action(self, parameters):
        (logits, mu, sigma) = parameters
        logit_shape = 8
        gaussian_shape = 1
        out_cat = tf.one_hot(tf.argmax(logits, axis=1), logit_shape)
        out_gauss = mu
        return tf.concat([out_cat, out_gauss], axis=1)

    def transform_action_sample(self, action_sample):
        out_cat, out_gauss = action_sample[:, :8], action_sample[:, 8:]
        print(out_cat, out_gauss)
        return tf.concat([out_cat, tf.tanh(out_gauss)], axis=1)


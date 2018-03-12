import tensorflow as tf
import numpy as np
from abc import abstractmethod

def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha*x)

class MLPPolicy(object):

    def input_processing(self, s):
        fc1 = tf.layers.dense(s, 128, tf.nn.relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 128, tf.nn.relu, name='fc2')
        return fc2


class GaussianPolicy(object):
    '''
    Policy outputs a gaussian action that is clamped to the interval [-1, 1]
    '''

    def produce_policy_parameters(self, a_shape, processed_s):
        mu_params = tf.layers.dense(processed_s, a_shape, name='mu_params')
        log_sigma_params = tf.layers.dense(processed_s, a_shape, name='sigma_params')
        return (mu_params, log_sigma_params)

    def policy_parameters_to_log_prob(self, a, parameters):
        (mu, log_sigma) = parameters
        # TODO confirm that this function behaves as expected.
        u = tf.atanh(a)
        log_prob = tf.distributions.Normal(mu, tf.exp(log_sigma)).log_prob(u)
        print(log_prob)
        return tf.reduce_sum(log_prob, axis=1) - tf.reduce_sum(tf.log(1 - tf.square(tf.tanh(u))), axis=1)

    def policy_parameters_to_sample(self, parameters):
        (mu, log_sigma) = parameters
        # TODO same here: confirm that this function behaves as expected
        # apply tanh to output of sample.
        return 0.99*tf.tanh(tf.distributions.Normal(mu, tf.exp(log_sigma)).sample())

class GaussianPolicy2(object):
    '''
    Policy outputs a gaussian action that is clamped to the interval [-1, 1]
    '''

    def produce_policy_parameters(self, a_shape, processed_s):
        mu_params = tf.layers.dense(processed_s, a_shape, name='mu_params')
        log_sigma_params = tf.layers.dense(processed_s, a_shape, name='sigma_params')
        return (mu_params, tf.minimum(log_sigma_params, 2))

    def policy_parameters_to_log_prob(self, a, parameters):
        (mu, log_sigma) = parameters
        # TODO confirm that this function behaves as expected.
        #a = tf.Print(a, [a], message='a', summarize=10)
        u = tf.atanh(a)
        #u = a
        #log_sigma = tf.Print(log_sigma, [log_sigma], message='log_sigma', summarize=10)
        #u = tf.Print(u, [u], message='u', summarize=10)
        quadratic = -0.5*tf.reduce_sum(tf.square(tf.exp(-log_sigma) * (u - mu)), axis=1)
        #quadratic = tf.Print(quadratic, [quadratic], message='quadratic', summarize=10)
        log_z = tf.reduce_sum(log_sigma, axis=1)
        D_t = tf.cast(tf.shape(mu)[1], tf.float32)
        log_z += 0.5*D_t*np.log(2*np.pi)
        log_p_no_correction = quadratic - log_z
        log_p = log_p_no_correction - tf.reduce_sum(tf.log(1 - tf.square(tf.tanh(u))), axis=1)
        #log_p = tf.Print(log_p, [log_p], message='Log-p', summarize=10)
        return log_p


        #log_prob = tf.distributions.Normal(mu, sigma).log_prob(u)
        #print(log_prob)
        #return tf.reduce_sum(log_prob, axis=1) - tf.reduce_sum(tf.log(1 - tf.square(tf.tanh(u))), axis=1)

    def policy_parameters_to_sample(self, parameters):
        (mu, log_sigma) = parameters
        # TODO same here: confirm that this function behaves as expected
        # apply tanh to output of sample.
        u_sample = mu + tf.exp(log_sigma) * tf.random_normal([tf.shape(mu)[0], mu.get_shape()[1].value])
        a_sample = tf.clip_by_value(tf.tanh(u_sample), -0.99, 0.99)
        return a_sample

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
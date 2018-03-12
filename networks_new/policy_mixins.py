import tensorflow as tf
from abc import abstractmethod

class AbstractPolicy(object):

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
            print(s)
            processed_s = self.input_processing(s)
            parameters = self.produce_policy_parameters(a_shape, processed_s)
            sample = self.policy_parameters_to_sample(parameters)
        return sample


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
        sigma_params = tf.layers.dense(processed_s, a_shape, tf.nn.sigmoid, name='sigma_params')
        return (mu_params, sigma_params)

    def policy_parameters_to_log_prob(self, a, parameters):
        (mu, sigma) = parameters
        # TODO confirm that this function behaves as expected.
        u = tf.atanh(a)
        log_prob = tf.distributions.Normal(mu, sigma).log_prob(u)
        return tf.reduce_sum(log_prob, axis=1) - tf.reduce_sum(tf.log(1 - tf.square(tf.tanh(u))), axis=1)


    def policy_parameters_to_sample(self, parameters):
        (mu, sigma) = parameters
        # TODO same here: confirm that this function behaves as expected
        # apply tanh to output of sample.
        return tf.tanh(tf.distributions.Normal(mu, sigma).sample())


class DiscretePolicy(object):


    def produce_policy_parameters(self, a_shape, processed_s):
        pass

    def policy_parameters_to_log_prob(self, a, parameters):
        pass

    def policy_parameters_to_sample(self, parameters):
        pass




# TODO build convolutional policy adapter
#class ConvPolicy(AbstractPolicy):
#
#    def input_processing(self, s):






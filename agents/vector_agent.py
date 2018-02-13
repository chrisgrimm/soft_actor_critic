import tensorflow as tf
from networks.actor_network import ActorNetwork
from networks.critic_network import CriticValueNetwork
import numpy as np


class VectorAgent(object):

    def __init__(self, state_size, action_size, hidden_size=64, num_gaussians=4):

        self.s1 = tf.placeholder(tf.float32, [None, state_size], name='state1')
        self.a = tf.placeholder(tf.float32, [None, action_size], name='action')
        self.s2 = tf.placeholder(tf.float32, [None, state_size], name='state2')
        self.r = tf.placeholder(tf.float32, [None], name='reward')
        self.t = tf.placeholder(tf.float32, [None], name='terminal')
        self.gamma = 0.001
        self.tau = 0.01
        self.learning_rate = 0.001

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians

        self.build_network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def build_network(self):
        self.actor_net = ActorNetwork(hidden_size=self.hidden_size, num_gaussians=self.num_gaussians)
        self.critic_value_net = CriticValueNetwork(hidden_size=self.hidden_size)

        V_xi = self.critic_value_net.V(self.s1, 'V_xi')
        Q_theta = self.critic_value_net.Q(self.s1, self.a, 'Q_theta')
        log_pi_phi = self.actor_net.log_pi(self.s1, self.a, 'pi_phi')
        V_xi_bar = self.critic_value_net.V(self.s2, 'V_xi_bar')

        xi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='V_xi')
        theta_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q_theta')
        phi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi_phi')
        xi_bar_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='V_xi_bar')



        self.J_V = 0.5*tf.reduce_mean(tf.square(V_xi  - tf.stop_gradient(tf.reduce_mean(Q_theta - log_pi_phi))))
        self.J_Q = 0.5*tf.reduce_mean(tf.square(Q_theta - tf.stop_gradient((self.r + (1 - self.t) * self.gamma*tf.reduce_mean(V_xi_bar)))))
        self.J_pi = log_pi_phi * tf.stop_gradient(log_pi_phi - Q_theta + V_xi)


        self.value_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.J_V, var_list=xi_vars)
        self.critic_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.J_Q, var_list=theta_vars)
        self.actor_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.J_pi, var_list=phi_vars)

        self.update_xi_bar = tf.group([tf.assign(xi_bar, self.tau*xi + (1 - self.tau)*xi_bar)
                                  for xi, xi_bar in zip(xi_vars, xi_bar_vars)])



    def act(self, state):
        [ws, mus, log_sigmas] = self.sess.run([self.actor_net.ws, self.actor_net.mus, self.actor_net.log_sigmas],
                                               feed_dict={self.s1: [state]})
        ws = [w[0] for w in ws]
        Z = np.sum(ws)
        ws /= Z
        mus = [mu[0] for mu in mus]
        log_sigmas = [log_sig[0] for log_sig in log_sigmas]
        w_choice = np.random.choice(range(len(ws)), p=ws)
        return np.random.normal(mus[w_choice], np.exp(log_sigmas[w_choice]))


    def train_step(self, s1, a, r, s2, t):
        [_1, _2, _3, JV, JQ, Jpi] = self.sess.run([self.value_opt, self.critic_opt, self.actor_opt,
                                                   self.J_V, self.J_Q, self.J_pi], feed_dict={self.s1: s1, self.a: a, self.r: r, self.s2: s2, self.t: t})
        return JV, JQ, Jpi





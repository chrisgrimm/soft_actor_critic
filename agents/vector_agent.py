import tensorflow as tf
from networks.actor_network import ActorNetwork
from networks.critic_network import CriticValueNetwork
import numpy as np
from shutil import rmtree

class VectorAgent(object):

    def __init__(self, state_size, action_size, hidden_size=128, num_gaussians=4, tb_dir=None, is_continuous=True):
        assert len(state_size) == 1
        try:
            rmtree(tb_dir)
        except FileNotFoundError:
            pass
        self.s1 = tf.placeholder(tf.float32, [None, state_size[0]], name='state1')
        self.a = tf.placeholder(tf.float32, [None, action_size[0]], name='action')
        self.a_sampled = tf.placeholder(tf.float32, [None, action_size[0]], name='action_sampled')
        self.s2 = tf.placeholder(tf.float32, [None, state_size[0]], name='state2')
        self.r = tf.placeholder(tf.float32, [None], name='reward')
        self.t = tf.placeholder(tf.float32, [None], name='terminal')
        self.global_step = 0
        self.gamma = 0.99
        self.tau = 0.01
        self.learning_rate = 3*10**-4
        self.is_continuous = is_continuous

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians

        self.build_network()

        self.summary_op = tf.summary.merge_all()
        if tb_dir is not None:
            self.tb_writer = tf.summary.FileWriter(tb_dir)
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.copy_xi_bar)


    def build_network(self):
        self.actor_net = ActorNetwork(action_shape=self.action_size, hidden_size=self.hidden_size, num_gaussians=self.num_gaussians, is_continuous=self.is_continuous)
        self.critic_value_net = CriticValueNetwork(hidden_size=self.hidden_size)

        V_xi = self.critic_value_net.V(self.s1, 'V_xi')
        Q_theta = self.critic_value_net.Q(self.s1, self.a, 'Q_theta')
        Q_theta_sampled = self.critic_value_net.Q(self.s1, self.a_sampled, 'Q_theta', reuse=True)
        log_pi_phi_sampled = self.actor_net.log_pi(self.s1, self.a_sampled, 'pi_phi')
        V_xi_bar = self.critic_value_net.V(self.s2, 'V_xi_bar')

        xi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='V_xi/')
        theta_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q_theta/')
        phi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi_phi/')
        xi_bar_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='V_xi_bar/')

        #print(xi_vars)
        #input('...')
        #print(theta_vars)
        #input('...')
        #print(phi_vars)
        #input('...')
        #print(xi_bar_vars)
        #input('...')

        self.J_V = 0.5*tf.reduce_mean(tf.square(V_xi - tf.stop_gradient(Q_theta_sampled - log_pi_phi_sampled)))

        self.J_Q = 0.5*tf.reduce_mean(tf.square(Q_theta - tf.stop_gradient((self.r + (1 - self.t) * self.gamma*V_xi_bar))))
        self.J_pi = tf.reduce_mean(log_pi_phi_sampled * tf.stop_gradient(log_pi_phi_sampled - Q_theta_sampled + V_xi))
        tf.summary.scalar('V loss', self.J_V)
        tf.summary.scalar('Q loss', self.J_Q)
        tf.summary.scalar('pi loss', self.J_pi)


        self.value_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.J_V, var_list=xi_vars)
        self.critic_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.J_Q, var_list=theta_vars)
        self.actor_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.J_pi, var_list=phi_vars)

        self.update_xi_bar = tf.group(*[tf.assign(xi_bar, self.tau*xi + (1 - self.tau)*xi_bar)
                                        for xi, xi_bar in zip(xi_vars, xi_bar_vars)])
        self.copy_xi_bar = tf.group(*[tf.assign(xi_bar, xi) for xi, xi_bar in zip(xi_vars, xi_bar_vars)])
        self.check_op = tf.add_check_numerics_ops()


    def act_continuous(self, states):
        [ws, mus, sigmas] = self.sess.run([self.actor_net.ws, self.actor_net.mus, self.actor_net.sigmas],
                                               feed_dict={self.s1: states})

        samples = []
        for i in range(len(states)):
            ws_i = [w[i][0] for w in ws]
            Z = np.sum(ws_i)
            ws_i /= Z
            mus_i = [mu[i] for mu in mus]
            sigmas_i = [sig[i] for sig in sigmas]
            w_choice = np.random.choice(range(len(ws_i)), p=ws_i)
            #print(mus_i[w_choice])
            #print(sigmas_i[w_choice])
            samples.append(np.tanh(np.random.normal(mus_i[w_choice], sigmas_i[w_choice])))
        return samples

    def act_discrete(self, states):
        [probs] = self.sess.run([self.actor_net.probs], feed_dict={self.s1: states})
        sampled_actions = []
        for i in range(len(states)):
            choice_idx = np.random.choice(range(self.action_size[0]), p=probs[i])
            action = np.zeros([self.action_size[0]])
            action[choice_idx] = 1
            sampled_actions.append(action)
        return sampled_actions


    def act(self, states):
        if self.is_continuous:
            return self.act_continuous(states)
        else:
            return self.act_discrete(states)

    def train_step(self, s1, a, r, s2, t):
        sampled_actions = self.act(s1)
        [summary, _1, _2, _3, JV, JQ, Jpi] = self.sess.run([self.summary_op, self.value_opt, self.critic_opt, self.actor_opt, self.J_V, self.J_Q, self.J_pi], feed_dict={self.s1: s1, self.a: a, self.s2: s2, self.t: t, self.r: r, self.a_sampled: sampled_actions})
        if self.tb_writer is not None:
            self.tb_writer.add_summary(summary, self.global_step)
        #self.tb_writer.add_summary(summary2, self.global_step)
        #self.tb_writer.add_summary(summary3, self.global_step)
        self.global_step += 1

        self.sess.run(self.update_xi_bar)
        return JV, JQ, Jpi





import numpy as np
from collections import namedtuple
from gym import spaces
from mujoco import ObjType
from os.path import join
import tensorflow as tf

from environments.base import at_goal
from environments.mujoco import MujocoEnv
from environments.pick_and_place import PickAndPlaceEnv
from sac.agent import mlp
from sac.replay_buffer import ReplayBuffer
from sac.utils import Step


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array(
        [
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
        ],
        dtype=np.float64)


Goal = namedtuple('Goal', 'gripper block')


class UnsupervisedEnv(PickAndPlaceEnv):
    def __init__(self, batch_size: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.sess = self.buffer = self.loss = self.train \
            = self.S1 = self.S2 = self.A = self.T = None
        self.state_size = [self.observation_space.shape[0] * 2] + \
                          list(self.observation_space.shape[1:])
        self._grad_clip = 1e6

    def initialize(self, session: tf.Session(), buffer: ReplayBuffer):
        with tf.variable_scope('env'):
            self.buffer = buffer
            self.sess = session

            self.S1 = tf.placeholder(tf.float32, [None] + self.state_size, name='S1')
            self.S2 = tf.placeholder(tf.float32, [None] + self.state_size, name='S2')
            self.A = tf.placeholder(tf.float32, [None] + list(self.action_space.shape), name='A')
            self.T = tf.placeholder(tf.float32, [None, 1], name='T')
            gamma = 0.99

            def network_output(inputs, name, reuse):
                with tf.variable_scope(name, reuse=reuse):
                    return mlp(inputs, layer_size=256, n_layers=3, activation=tf.nn.relu)

            v1 = network_output(self.S1, name='Q', reuse=False)
            v2 = network_output(self.S2, name='Q', reuse=True)
            q1 = v1 + network_output(tf.concat([self.S1, self.A], axis=1), name='A', reuse=False)
            self.loss = tf.reduce_mean(0.5 * tf.square(q1 - gamma * (self.T + (1 - self.T) * v2)))
            optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            if self._grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, self._grad_clip)
            self.train = optimizer.apply_gradients(zip(gradients, variables))
            session.run(tf.global_variables_initializer())

    def compute_terminal(self, goal, obs):
        return False

    def _block_height(self):
        return self.block_pos()[-1] - self._initial_block_pos[-1]

    def goal(self):
        pass

    def compute_reward(self, goal, obs):
        assert goal is None
        if self.train is None:
            raise RuntimeError("Need to run `UnsupervisedEnv.initialize` first.")
        if self.buffer.empty:
            return 0
        sample_steps = Step(*self.buffer.sample(self.batch_size))
        goal, = obs  # use current obs as goal
        s1 = self._add_goal_to_sample(obs=sample_steps.s1, goal=goal)
        s2 = self._add_goal_to_sample(obs=sample_steps.s2, goal=goal)
        t = at_goal(sample_steps.s2, goal, self._geofence)
        return self.sess.run([self.loss, self.train],
                             feed_dict={
                                 self.S1: s1,
                                 self.S2: s2,
                                 self.A: sample_steps.a,
                                 self.T: np.reshape(t, (self.batch_size, 1))
                             })[0]

    def step(self, action):
        s, r, t, i = super().step(action)
        if self._block_height() > self._min_lift_height:
            i['print'] = 'Block lifted by {}. Reward: {}'.format(self._block_height(), r)
            i['log'] = {'block-lifted': 1}
        else:
            i['log'] = {'block-lifted': 0}
        return self._vectorize_obs(s), r, t, i

    def reset(self):
        reset = super().reset()
        return self._vectorize_obs(reset)

    @staticmethod
    def _add_goal_to_sample(obs, goal):
        return [np.concatenate([o, goal]) for o in obs]

    @staticmethod
    def _vectorize_obs(obs):
        return np.concatenate([q for q, in obs])

"""Create gym environment for HSR"""

from collections import deque

import gym
import numpy as np
from gym import utils

from environment.server import Server
from abc import abstractmethod


class BaseEnv(utils.EzPickle, Server):
    """ The environment """

    def __init__(self, max_steps, history_len, image_dimensions,
                 neg_reward, steps_per_action):
        utils.EzPickle.__init__(self)

        self._history_buffer = deque(maxlen=history_len)
        self._steps_per_action = steps_per_action
        self._step_num = 0
        self._neg_reward = neg_reward
        self._image_dimensions = image_dimensions
        self.max_steps = max_steps

        self._history_buffer += [self._obs()] * history_len
        self.observation_space = self.action_space = None

        # required for OpenAI code
        self.metadata = {'render.modes': 'rgb_array'}
        self.reward_range = -np.inf, np.inf
        self.spec = None

    def mlp_input(self, goal, history):
        assert len(history) > 0
        obs_history = [np.concatenate(x, axis=0) for x in history]
        return np.concatenate(list(goal) + obs_history, axis=0)

    def destructure_mlp_input(self, mlp_input):
        assert isinstance(self.observation_space, gym.Space)
        assert self.observation_space.contains(mlp_input)
        goal_shapes = [np.size(x) for x in self._goal()]
        goal_size = sum(goal_shapes)

        # split mlp_input into goal and obs pieces
        goal_vector, obs_history = mlp_input[:goal_size], mlp_input[goal_size:]

        history_len = len(self._history_buffer)
        assert np.size(goal_vector) == goal_size
        assert (np.size(obs_history)) % history_len == 0

        # break goal vector into individual goals
        goals = np.split(goal_vector, np.cumsum(goal_shapes), axis=0)[:-1]

        # break history into individual observations in history
        history = np.split(obs_history, history_len, axis=0)

        obs_shapes = [np.size(x) for x in self._obs()]
        obs = []

        # break each observation in history into observation pieces
        for o in history:
            assert np.size(o) == sum(obs_shapes)
            obs.append(np.split(o, np.cumsum(obs_shapes), axis=0)[:-1])

        return goals, obs

    def step(self, action):
        self._step_num += 1
        step = 0
        reward = 0
        done = False

        while not done and step < self._steps_per_action:
            self._perform_action(action)
            hit_max_steps = self._step_num >= self.max_steps
            done = False
            if self._compute_terminal(self._goal(), self._obs()):
                # print('terminal')
                done = True
            elif hit_max_steps:
                # print('hit max steps')
                done = True
            elif self._currently_failed():
                done = True
            reward += self._compute_reward(self._goal(), self._obs())
            step += 1

        self._history_buffer.append(self._obs())
        return self._history_buffer, reward, done, {}


    @staticmethod
    def seed(seed):
        np.random.seed(seed)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @abstractmethod
    def _perform_action(self, action):
        raise NotImplemented

    @abstractmethod
    def render(self, mode=None, camera_name=None, labels=None):
        raise NotImplemented

    @abstractmethod
    def image(self, camera_name='rgb'):
        raise NotImplemented

    @abstractmethod
    def _step_inner(self, action):
        raise NotImplemented

    @abstractmethod
    def reset(self):
        raise NotImplemented

    @abstractmethod
    def _set_new_goal(self):
        raise NotImplemented

    @abstractmethod
    def _obs(self):
        raise NotImplemented

    @abstractmethod
    def _goal(self):
        raise NotImplemented

    @abstractmethod
    def goal_3d(self):
        raise NotImplemented

    @abstractmethod
    def _currently_failed(self):
        raise NotImplemented

    @abstractmethod
    def _compute_terminal(self, goal, obs):
        raise NotImplemented

    @abstractmethod
    def _compute_reward(self, goal, obs):
        raise NotImplemented

    # hindsight stuff
    def _obs_to_goal(self, obs):
        raise NotImplemented

    def obs_to_goal(self, mlp_input):
        goal, obs_history = self.destructure_mlp_input(mlp_input)
        return self._obs_to_goal(obs_history[-1])

    def change_goal(self, goal, mlp_input):
        _, obs_history = self.destructure_mlp_input(mlp_input)
        return self.mlp_input(goal, obs_history)

    def compute_reward(self, mlp_input):
        goal, obs_history = self.destructure_mlp_input(mlp_input)
        return sum(self._compute_reward(goal, obs) for obs in obs_history)

    def compute_terminal(self, mlp_input):
        goal, obs_history = self.destructure_mlp_input(mlp_input)
        return any(self._compute_terminal(goal, obs) for obs in obs_history)


def quaternion2euler(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    euler_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = 1 if t2 > 1 else t2
    t2 = -1 if t2 < -1 else t2
    euler_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    euler_z = np.arctan2(t3, t4)

    return euler_x, euler_y, euler_z


def distance_between(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2)))


def at_goal(pos, goal, geofence):
    distance_to_goal = distance_between(pos, goal)
    return distance_to_goal < geofence


def escaped(pos, world_upper_bound, world_lower_bound):
    # noinspection PyTypeChecker
    return np.any(pos > world_upper_bound) \
           or np.any(pos < world_lower_bound)


def get_limits(pos, size):
    return pos + size, pos - size


def point_inside_object(point, object):
    pos, size = object
    tl = pos - size
    br = pos + size
    return (tl[0] <= point[0] <= br[0]) and (tl[1] <= point[1] <= br[1])


def print1(*strings):
    print('\r', *strings, end='')

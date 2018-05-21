"""Create gym environment for HSR"""
from abc import abstractmethod
from collections import deque
from copy import deepcopy

import gym
import numpy as np


class BaseEnv(gym.Env):
    """ The environments """

    def __init__(self, history_len, image_dimensions, neg_reward,
                 steps_per_action):

        self._history_buffer = deque(maxlen=history_len)
        self._steps_per_action = steps_per_action
        self._step_num = 0
        self._neg_reward = neg_reward
        self._image_dimensions = image_dimensions

        self._history_buffer += [self._obs()] * history_len
        self.observation_space = self.action_space = None

        # required for OpenAI code
        self.metadata = {'render.modes': 'rgb_array'}
        self.reward_range = -np.inf, np.inf
        self.spec = None

    def step(self, action):
        self._step_num += 1
        step = 0
        reward = 0
        done = False

        while not done and step < self._steps_per_action:
            self._perform_action(action)
            done = False
            if self.compute_terminal(self.goal(), self._obs()):
                done = True
            elif self._currently_failed():
                done = True
            reward += self.compute_reward(self.goal(), self._obs())
            step += 1

        self._history_buffer.append(self._obs())
        return deepcopy(self._history_buffer), reward, done, {}

    def seed(self, seed=None):
        np.random.seed(seed)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @abstractmethod
    def _perform_action(self, action):
        raise NotImplementedError

    @abstractmethod
    def render(self, mode=None, camera_name=None, labels=None):
        raise NotImplementedError

    @abstractmethod
    def image(self, camera_name='rgb'):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def _set_new_goal(self):
        raise NotImplementedError

    @abstractmethod
    def _obs(self):
        raise NotImplementedError

    @abstractmethod
    def goal(self):
        raise NotImplementedError

    @abstractmethod
    def goal_3d(self):
        raise NotImplementedError

    @abstractmethod
    def _currently_failed(self):
        raise NotImplementedError

    @abstractmethod
    def compute_terminal(self, goal, obs):
        raise NotImplementedError

    @abstractmethod
    def compute_reward(self, goal, obs):
        raise NotImplementedError


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
    return np.sqrt(np.sum(np.square(pos1 - pos2), axis=-1))


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

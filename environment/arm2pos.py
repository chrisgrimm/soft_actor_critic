from os.path import join

import numpy as np
from gym import spaces

from environment.base import at_goal, BaseEnv
from environment.mujoco import MujocoEnv


class Arm2PosEnv(MujocoEnv):
    def __init__(self, continuous, max_steps, history_len=1, neg_reward=True,
                 action_multiplier=1):

        super().__init__(
                         max_steps=max_steps,
                         xml_filepath=join('models', 'arm2pos', 'world.xml'),
                         history_len=history_len,
                         neg_reward=neg_reward,
                         steps_per_action=10,
                         image_dimensions=None)

        left_finger_name = 'hand_l_distal_link'
        self._finger_names = [left_finger_name, left_finger_name.replace('_l_', '_r_')]
        self._set_new_goal()
        self._action_multiplier = action_multiplier
        self._continuous = continuous
        obs_shape = history_len * np.size(self._obs()) + np.size(self._goal())
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape)

        if continuous:
            self.action_space = spaces.Box(-1, 1, shape=self.sim.nu)
        else:
            self.action_space = spaces.Discrete(self.sim.nu * 2 + 1)

    def reset_qpos(self):
        return self.init_qpos

    def _set_new_goal(self):
        low = np.array([-0.27217179, -0.17194216,  0.50806907])
        high = np.array([0.11727834, 0.32794162, 0.50806907])
        goal = np.random.uniform(low, high)
        assert np.all(low <= goal) and np.all(goal <= high)
        self.__goal = goal

    def _obs(self):
        return [self.sim.qpos]

    def _goal(self):
        return [self.__goal]

    def goal_3d(self):
        return self.__goal

    def _currently_failed(self):
        return False

    def _compute_terminal(self, goal, obs):
        goal, = goal
        qpos, = obs
        return at_goal(self._gripper_pos(qpos), goal, self._geofence)

    def _compute_reward(self, goal, obs):
        goal_pos, = goal
        qpos, = obs
        if at_goal(self._gripper_pos(qpos), goal_pos, self._geofence):
            return 1
        elif self._neg_reward:
            return -.0001
        else:
            return 0

    def _obs_to_goal(self, obs):
        qpos, = obs
        return [self._gripper_pos(qpos)]

    def _gripper_pos(self, qpos=None):
        finger1, finger2 = [self.sim.get_body_xpos(name, qpos)
                            for name in self._finger_names]
        return (finger1 + finger2) / 2.

    def step(self, action):
        if not self._continuous:
            ctrl = np.zeros(self.sim.nu)
            if action != 0:
                ctrl[(action - 1) // 2] = (1 if action % 2 else -1) * self._action_multiplier
            return BaseEnv.step(self, ctrl)
        else:
            action = np.clip(action * self._action_multiplier, -1, 1)
            return BaseEnv.step(self, action)

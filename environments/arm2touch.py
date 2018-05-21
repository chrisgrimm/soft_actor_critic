from os.path import join

import numpy as np
from gym import spaces

from environments.base import BaseEnv


class Arm2TouchEnv(BaseEnv):
    def __init__(self,
                 continuous,
                 max_steps,
                 geofence=.08,
                 history_len=1,
                 neg_reward=True,
                 action_multiplier=1):

        BaseEnv.__init__(
            self,
            geofence=geofence,
            max_steps=max_steps,
            xml_filepath=join('models', 'arm2touch', 'world.xml'),
            history_len=history_len,
            use_camera=False,  # TODO
            neg_reward=neg_reward,
            body_name="hand_palm_link",
            steps_per_action=10,
            image_dimensions=None)

        left_finger_name = 'hand_l_distal_link'
        self._finger_names = [
            left_finger_name,
            left_finger_name.replace('_l_', '_r_')
        ]
        self._set_new_goal()
        self._action_multiplier = action_multiplier
        self._continuous = continuous
        obs_shape = history_len * np.size(self._obs()) + np.size(self.goal())
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape)

        if continuous:
            self.action_space = spaces.Box(-1, 1, shape=self.sim.nu)
        else:
            self.action_space = spaces.Discrete(self.sim.nu * 2 + 1)

    def generate_valid_block_position(self):
        low_range = np.array([-0.15, -0.25, 0.49])
        high_range = np.array([0.15, 0.25, 0.49])
        return np.random.uniform(low=low_range, high=high_range)

    def get_block_position(self, qpos, name):
        idx = self.sim.jnt_qposadr(name)
        position = qpos[idx:idx + 3]
        return np.copy(position)

    def set_block_position(self, qpos, name, position):
        idx = self.sim.jnt_qposadr(name)
        qpos = np.copy(qpos)
        qpos[idx:idx + 3] = position
        return qpos

    def are_positions_touching(self, pos1, pos2):
        touching_threshold = 0.05
        weighting = np.array([1, 1, 0.1])
        dist = np.sqrt(np.sum(weighting * np.square(pos1 - pos2)))
        return dist < touching_threshold

    def reset_qpos(self):
        qpos = self.init_qpos
        qpos = self.set_block_position(self.sim.qpos, 'block1joint',
                                       self.generate_valid_block_position())
        qpos = self.set_block_position(self.sim.qpos, 'block2joint',
                                       self.generate_valid_block_position())
        return qpos

    def _set_new_goal(self):
        goal_block = np.random.randint(0, 2)
        onehot = np.zeros([2], dtype=np.float32)
        onehot[goal_block] = 1
        self.__goal = onehot

    def _obs(self):
        return [self.sim.qpos]

    def goal(self):
        return [self.__goal]

    def goal_3d(self):
        return [0, 0, 0]

    def _currently_failed(self):
        return False

    def at_goal(self, qpos, goal):
        block1 = self.get_block_position(qpos, 'block1joint')
        block2 = self.get_block_position(qpos, 'block2joint')
        gripper = self._gripper_pos(qpos)
        goal_block = np.argmax(goal) + 1
        if goal_block == 1:
            return self.are_positions_touching(block1, gripper)
        else:
            return self.are_positions_touching(block2, gripper)

    def compute_terminal(self, goal, obs):
        goal, = goal
        qpos, = obs
        return self.at_goal(qpos, goal)

    def compute_reward(self, goal, obs):
        qpos, = obs
        if self.at_goal(qpos, goal):
            return 1
        elif self._neg_reward:
            return -.0001
        else:
            return 0

    def _obs_to_goal(self, obs):
        raise Exception('No promises here.')
        qpos, = obs
        return [self.gripper_pos(qpos)]

    def _gripper_pos(self, qpos=None):
        finger1, finger2 = [
            self.sim.get_body_xpos(name, qpos) for name in self._finger_names
        ]
        return (finger1 + finger2) / 2.

    def step(self, action):
        if not self._continuous:
            ctrl = np.zeros(self.sim.nu)
            if action != 0:
                ctrl[(action - 1) // 2] = (
                    1 if action % 2 else -1) * self._action_multiplier
            return BaseEnv.step(self, ctrl)
        else:
            action = np.clip(action * self._action_multiplier, -1, 1)
            return BaseEnv.step(self, action)

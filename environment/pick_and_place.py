from collections import namedtuple
from os.path import join

import numpy as np
from gym import spaces
from mujoco import ObjType

from environment.base import at_goal
from environment.mujoco import MujocoEnv


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


def failed(resting_block_height, goal_block_height):
    return False


Goal = namedtuple('Goal', 'gripper block')


class PickAndPlaceEnv(MujocoEnv):
    def __init__(self,
                 max_steps,
                 min_lift_height=.02,
                 geofence=.06,
                 neg_reward=False,
                 history_len=1):
        self._goal_block_name = 'block1'
        self._min_lift_height = min_lift_height + geofence
        self._geofence = geofence

        super().__init__(
            max_steps=max_steps,
            xml_filepath=join('models', 'pick-and-place', 'world.xml'),
            history_len=history_len,
            neg_reward=neg_reward,
            steps_per_action=20,
            image_dimensions=None)

        self.initial_qpos = np.copy(self.init_qpos)
        self._initial_block_pos = np.copy(self.block_pos())
        left_finger_name = 'hand_l_distal_link'
        self._finger_names = [
            left_finger_name,
            left_finger_name.replace('_l_', '_r_')
        ]
        obs_size = history_len * sum(map(np.size, self._obs())) + sum(
            map(np.size, self.goal()))
        assert obs_size != 0
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_size, ), dtype=np.float32)
        self.action_space = spaces.Box(
            -1, 1, shape=(self.sim.nu - 1, ), dtype=np.float32)
        self._table_height = self.sim.get_body_xpos('pan')[2]
        self._rotation_actuators = ["arm_flex_motor"]  # , "wrist_roll_motor"]

        # self._n_block_orientations = n_orientations = 8
        # self._block_orientations = np.random.uniform(0, 2 * np.pi,
        # size=(n_orientations, 4))
        # self._rewards = np.ones(n_orientations) * -np.inf
        # self._usage = np.zeros(n_orientations)
        # self._current_orienation = None

    def reset_qpos(self):
        # self.init_qpos = np.array([4.886e-05,
        #                            - 2.152e-05,
        #                            4.385e-01,
        #                            1.000e+00,
        #                            2.254e-17,
        #                            - 2.388e-19,
        #                            1.290e-05,
        #                            - 9.773e-01,
        #                            2.773e-02,
        #                            3.573e-01,
        #                            3.574e-01, ])
        # if np.random.uniform(0, 1) < .5:
        #     self.init_qpos = np.array([
        #         7.450e-05,
        #         -3.027e-03,
        #         4.385e-01,
        #         1.000e+00,
        #         0,
        #         0,
        #         -6.184e-04,
        #         -1.101e+00,
        #         0,
        #         3.573e-01,
        #         3.574e-01,
        #     ])
        # else:
        #     self.init_qpos = self.initial_qpos

        block_joint = self.sim.jnt_qposadr('block1joint')

        self.init_qpos[block_joint + 3] = np.random.uniform(0, 1)
        self.init_qpos[block_joint + 6] = np.random.uniform(-1, 1)

        # self.init_qpos[block_joint + 3:block_joint + 7] = np.random.random(
        #     4) * 2 * np.pi
        # rotate_around_x = [np.random.uniform(0, 1), np.random.uniform(-1, 1), 0, 0]
        # rotate_around_z = [np.random.uniform(0, 1), 0, 0, np.random.uniform(-1, 1)]
        # w, x, y, z = quaternion_multiply(rotate_around_z, rotate_around_x)
        # self.init_qpos[block_joint + 3] = w
        # self.init_qpos[block_joint + 4] = x
        # self.init_qpos[block_joint + 5] = y
        # self.init_qpos[block_joint + 6] = z
        # mean_rewards = self._rewards / np.maximum(self._usage, 1)
        # self._current_orienation = i = np.argmin(mean_rewards)
        # print('rewards:', mean_rewards, 'argmin:', i)
        # self._usage[i] += 1
        # self.init_qpos[block_joint + 3:block_joint + 7] = self._block_orientations[i]
        # self.init_qpos[self.sim.jnt_qposadr(
        #     'wrist_roll_joint')] = np.random.random() * 2 * np.pi
        return self.init_qpos

    def _set_new_goal(self):
        pass

    def _obs(self):
        return self.sim.qpos,

    def block_pos(self):
        return self.sim.get_body_xpos(self._goal_block_name)

    def goal(self):
        goal_pos = self._initial_block_pos + \
            np.array([0, 0, self._min_lift_height])
        return Goal(gripper=goal_pos, block=goal_pos)

    def goal_3d(self):
        return self.goal()[0]

    def _currently_failed(self):
        return False

    def _achieved_goal(self, goal, obs):
        gripper_at_goal = at_goal(
            self.gripper_pos(obs[0]), goal.gripper, self._geofence)
        block_at_goal = at_goal(self.block_pos(), goal.block, self._geofence)
        return gripper_at_goal and block_at_goal

    def compute_terminal(self, goal, obs):
        return self._achieved_goal(goal, obs)

    def compute_reward(self, goal, obs):
        if self._achieved_goal(goal, obs):
            print('Achieved goal')
            return 1
        elif self._neg_reward:
            return -.0001
        else:
            return 0

    def gripper_pos(self, qpos=None):
        finger1, finger2 = [
            self.sim.get_body_xpos(name, qpos) for name in self._finger_names
        ]
        return (finger1 + finger2) / 2.

    def step(self, action):
        action = np.clip(action, -1, 1)
        for name in self._rotation_actuators:
            i = self.sim.name2id(ObjType.ACTUATOR, name)
            action[i] *= np.pi / 2

        mirrored = [
            'hand_l_proximal_motor',
            # 'hand_l_distal_motor'
        ]
        mirroring = [
            'hand_r_proximal_motor',
            # 'hand_r_distal_motor'
        ]

        def get_indexes(names):
            return [self.sim.name2id(ObjType.ACTUATOR, name) for name in names]

        # insert mirrored values at the appropriate indexes
        mirrored_indexes, mirroring_indexes = map(get_indexes,
                                                  [mirrored, mirroring])
        # necessary because np.insert can't append multiple values to end:
        mirroring_indexes = np.minimum(mirroring_indexes,
                                       self.action_space.shape)
        action = np.insert(action, mirroring_indexes, action[mirrored_indexes])
        return super().step(action)

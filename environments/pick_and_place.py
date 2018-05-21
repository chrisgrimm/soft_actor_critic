from collections import namedtuple
from os.path import join

import numpy as np
from gym import spaces

from environments.base import at_goal, print1
from environments.mujoco import MujocoEnv
from mujoco import ObjType


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


class PickAndPlaceEnv(MujocoEnv):
    def __init__(self,
                 random_block,
                 min_lift_height=.02,
                 geofence=.04,
                 neg_reward=False,
                 history_len=1,
                 discrete=False):
        self.grip = 0
        self._random_block = random_block
        self._goal_block_name = 'block1'
        self._min_lift_height = min_lift_height + geofence
        self._geofence = geofence
        self._discrete = discrete

        super().__init__(
            xml_filepath=join('models', 'pick-and-place', 'discrete.xml'
                              if discrete else 'world.xml'),
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
        obs_size = history_len * sum(map(np.size, self._obs()))
        assert obs_size != 0
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_size, ), dtype=np.float32)
        if discrete:
            self.action_space = spaces.Discrete(7)
        else:
            self.action_space = spaces.Box(
                low=np.array([-15, -20, -20]), high=np.array([35, 20, 20]), dtype=np.float32)
        self._table_height = self.sim.get_body_xpos('pan')[2]
        self._rotation_actuators = ["arm_flex_motor"]  # , "wrist_roll_motor"]

        # self._n_block_orientations = n_orientations = 8
        # self._block_orientations = np.random.uniform(0, 2 * np.pi,
        # size=(n_orientations, 4))
        # self._rewards = np.ones(n_orientations) * -np.inf
        # self._usage = np.zeros(n_orientations)
        # self._current_orienation = None

    def reset_qpos(self):
        if self._random_block:
            block_joint = self.sim.jnt_qposadr('block1joint')

            self.init_qpos[block_joint + 3] = np.random.uniform(0, 1)
            self.init_qpos[block_joint + 6] = np.random.uniform(-1, 1)

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
        return np.copy(self.sim.qpos),

    def block_pos(self, qpos=None):
        return self.sim.get_body_xpos(self._goal_block_name, qpos)

    def gripper_pos(self, qpos=None):
        finger1, finger2 = [
            self.sim.get_body_xpos(name, qpos) for name in self._finger_names
        ]
        return (finger1 + finger2) / 2.

    def goal(self):
        goal_pos = self._initial_block_pos + \
            np.array([0, 0, self._min_lift_height])
        return Goal(gripper=goal_pos, block=goal_pos)

    def goal_3d(self):
        return self.goal()[0]

    def _currently_failed(self):
        return False

    def at_goal(self, goal, obs):
        qpos, = obs
        gripper_at_goal = at_goal(
            self.gripper_pos(qpos), goal.gripper, self._geofence)
        block_at_goal = at_goal(
            self.block_pos(qpos), goal.block, self._geofence)
        return gripper_at_goal and block_at_goal

    def compute_terminal(self, goal, obs):
        # return False
        return self.at_goal(goal, obs)

    def compute_reward(self, goal, obs):
        if self.at_goal(goal, obs):
            return 1
        elif self._neg_reward:
            return -.0001
        else:
            return 0

    def step(self, action):
        if self._discrete:
            a = np.zeros(4)
            if action > 0:
                action -= 1
                joint = action // 2
                assert 0 <= joint <= 2
                direction = (-1)**(action % 2)
                joint_scale = [.2, .05, .5]
                a[2] = self.grip
                a[joint] = direction * joint_scale[joint]
                self.grip = a[2]
            action = a
        action = np.clip(action, self.action_space.low, self.action_space.high)

        mirrored = 'hand_l_proximal_motor'
        mirroring = 'hand_r_proximal_motor'

        # insert mirrored values at the appropriate indexes
        mirrored_index, mirroring_index = [
            self.sim.name2id(ObjType.ACTUATOR, n)
            for n in [mirrored, mirroring]
        ]
        # necessary because np.insert can't append multiple values to end:
        if self._discrete:
            action[mirroring_index] = action[mirrored_index]
        else:
            mirroring_index = np.minimum(mirroring_index,
                                         self.action_space.shape)
            action = np.insert(action, mirroring_index, action[mirrored_index])
        return super().step(action)

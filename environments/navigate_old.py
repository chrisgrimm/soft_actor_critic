import os
from os.path import join

import numpy as np
from gym.spaces import Box, Discrete, Tuple

from environments.base import (BaseEnv, at_goal, escaped, point_inside_object,
                               quaternion2euler)
from mujoco import GeomType, ObjType


class NavigateEnv(BaseEnv):
    def __init__(self,
                 geofence=.3,
                 goal_log_file=None,
                 max_steps=6000,
                 history_len=4,
                 image_dimensions=(64, 64),
                 use_camera=True,
                 continuous_actions=False,
                 neg_reward=False,
                 steps_per_action=20,
                 action_multiplier=1):

        super().__init__(
            geofence=geofence,
            max_steps=max_steps,
            xml_filepath=join('models', 'navigate', 'world.xml'),
            image_dimensions=image_dimensions,
            use_camera=use_camera,
            neg_reward=neg_reward,
            body_name='base_link',
            history_len=history_len,
            steps_per_action=steps_per_action)

        self._floor_id = self.sim.name2id(ObjType.GEOM, 'floor')
        self._world_size = self.sim.get_geom_size(self._floor_id)[:2]
        self._world_offset = self.sim.get_geom_pos(self._floor_id)[:2]
        self._world_upper_bound = self._world_offset + self._world_size
        self._world_lower_bound = self._world_offset - self._world_size
        self._goal_log_file = goal_log_file
        self._continuous_actions = continuous_actions
        self._action_multiplier = action_multiplier
        self._body_radius = 0.25

        self._set_new_goal()

        if continuous_actions:
            self.action_space = Box(-1, 1, 3)
        else:
            self.action_space = Discrete(5)

        cnn_space = Box(
            0, 1, shape=(list(image_dimensions) + [3 * history_len]))
        obs_size = history_len * \
            sum(map(np.size, self._obs())) + sum(map(np.size, self.goal()))
        mlp_space = Box(
            np.min(self._world_lower_bound),
            np.min(self._world_upper_bound),
            shape=obs_size)

        if use_camera:
            self.observation_space = Tuple([mlp_space, cnn_space])
        else:
            self.observation_space = mlp_space

        # log positions
        self.log_start_pos = None
        if self._goal_log_file:
            try:
                os.remove(self._goal_log_file)
            except IOError:
                pass

    def server_values(self):
        return self.sim.qpos, self.sim.qvel, self.goal

    def reset_qpos(self):
        qpos = np.append(self._get_new_pos(), [0])
        self.log_start_pos = qpos
        return qpos

    def _set_new_goal(self):
        self.__goal = self._get_new_pos()

    def _obs(self):
        obs = [self._pos(), self._orientation()]
        if self._use_camera:
            obs += [self.image()]
        return obs

    def goal(self):
        return [self.__goal]

    def goal_3d(self):
        return np.append(self.__goal, 0)

    def _currently_failed(self):
        return escaped(self._pos(), self._world_upper_bound,
                       self._world_lower_bound)

    def _orientation(self):
        quat = self.sim.get_body_xquat(self._body_name)
        x, y, z = quaternion2euler(*quat)
        return np.array([np.cos(z), np.sin(z)])

    def _pos(self):
        return self.sim.get_body_xpos(self._body_name)[:2]

    def compute_terminal(self, goal, obs):
        goal, = goal
        pos = obs[0]
        return at_goal(pos, goal, self._geofence)

    def compute_reward(self, goal, obs):
        pos = obs[0]
        if at_goal(pos, goal, self._geofence):
            return 1
        elif escaped(pos, self._world_upper_bound, self._world_lower_bound):
            return -1
        elif self._neg_reward:
            return -0.01
        else:
            return 0

    def _obs_to_goal(self, obs):
        return [obs[0]]

    def step(self, action):
        if self._continuous_actions:
            action = np.clip(action, -1, 1)
            action[2] *= .01
        else:
            action_index = action
            action = [
                [0, 0],
                [-1, 0],
                [1, 0],
                [0, -1],
                [0, 1],
            ][action_index]

            forward, rotate = np.array(action)
            action = np.append(forward * self._orientation(), [rotate * .01])
        obs, reward, done, meta = super().step(
            action * self._action_multiplier)
        if done:
            self._write_log_file()
        return obs, reward, done, meta

    def at_goal(self, goal, new_obs):
        without_goal = new_obs[:-2]
        position_orientation = without_goal[-4:]
        position = position_orientation[:2]
        return at_goal(position, goal, self._geofence)

    def _get_new_pos(self, rel_to=None):
        while True:
            if rel_to is None:
                pos = np.random.uniform(self._world_lower_bound,
                                        self._world_upper_bound)

            else:
                coord, min_radius, max_radius = rel_to
                assert not self._intersects_object(coord)
                radius = np.random.uniform(min_radius, max_radius)
                angle = np.random.uniform(-np.pi, np.pi)
                offset = np.array([np.cos(angle), np.sin(angle)]) * radius
                pos = coord + offset

            world = self._world_offset, self._world_size - self._body_radius
            in_world = point_inside_object(pos, world)
            if in_world and not self._intersects_object(pos):
                return pos

    def _intersects_object(self, point):
        def get_geom_size(i):
            size = self.sim.get_geom_size(i)
            if self.sim.get_geom_type(i) == GeomType.CYLINDER:
                return size[0] * np.ones(2)
            return size[:2]

        objects = [(self.sim.get_geom_pos(i)[:2], get_geom_size(i))
                   for i in range(self.sim.nbody) if i is not self._floor_id]
        for obj in objects:
            if point_inside_object(point, obj):
                return True
        return False

    def _write_log_file(self):
        if self._goal_log_file is not None:
            values = np.concatenate([[self._step_num], self.log_start_pos,
                                     self.goal(),
                                     self._pos()])
            with open(self._goal_log_file, 'a') as f:
                f.write(' '.join(map(str, values)))

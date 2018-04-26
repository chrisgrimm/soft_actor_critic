import os
from abc import abstractmethod

import mujoco
import numpy as np

from environment.base import BaseEnv


class MujocoEnv(BaseEnv):
    def __init__(self, xml_filepath, max_steps, history_len, image_dimensions,
                 neg_reward, steps_per_action, frames_per_step=20):
        fullpath = os.path.join(os.path.dirname(__file__), xml_filepath)
        if not fullpath.startswith("/"):
            fullpath = os.path.join(os.path.dirname(__file__), "assets", fullpath)
        self.sim = mujoco.Sim(fullpath)
        self.init_qpos = self.sim.qpos.ravel().copy()
        self.init_qvel = self.sim.qvel.ravel().copy()
        self._frames_per_step = frames_per_step
        super().__init__(max_steps, history_len, image_dimensions,
                         neg_reward, steps_per_action)

    def server_values(self):
        return self.sim.qpos, self.sim.qvel

    def render(self, mode=None, camera_name=None, labels=None):
        if mode == 'rgb_array':
            return self.sim.render_offscreen(height=256, width=256)
        self.sim.render(camera_name, labels)

    def image(self, camera_name='rgb'):
        return self.sim.render_offscreen(
            *self._image_dimensions, camera_name)

    def step(self, action):
        assert np.shape(action) == np.shape(self.sim.ctrl)
        return super().step(action)

    def _perform_action(self, action):
        assert np.shape(action) == np.shape(self.sim.ctrl)
        self.sim.ctrl[:] = action
        for _ in range(self._frames_per_step):
            self.sim.step()

    def reset(self):
        self.sim.reset()
        self._step_num = 0

        self._set_new_goal()
        qpos = self.reset_qpos()
        qvel = self.init_qvel + \
               np.random.uniform(size=self.sim.nv, low=-0.01, high=0.01)
        assert qpos.shape == (self.sim.nq,) and qvel.shape == (self.sim.nv,)
        self.sim.qpos[:] = qpos.copy()
        self.sim.qvel[:] = qvel.copy()
        self.sim.forward()
        return self.mlp_input(self._goal(), self._history_buffer)

    @abstractmethod
    def reset_qpos(self):
        raise NotImplemented

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.sim.__exit__()


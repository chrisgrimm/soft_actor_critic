from abc import abstractmethod
from collections import namedtuple

import gym
import numpy as np

from environment.pick_and_place import PickAndPlaceEnv
from gym.spaces import Box

State = namedtuple('State', 'obs goal')


class GoalWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.trajectory = None
        self.current_state = None
        concatenate = self.concatenate(self.reset())
        self.observation_space = Box(-1, 1, concatenate.shape)

    @abstractmethod
    def goal_from_obs_part(self, obs_part):
        raise NotImplementedError

    @abstractmethod
    def reward(self, obs_part, goal):
        raise NotImplementedError

    @abstractmethod
    def terminal(self, obs_part, goal):
        raise NotImplementedError

    @abstractmethod
    def desired_goal(self):
        pass

    @staticmethod
    def concatenate(state):
        return np.concatenate(state)

    def step(self, action):
        s2, r, t, info = self.env.step(action)
        new_s2 = State(s2, self.desired_goal())
        new_r = self.reward(s2, self.desired_goal())
        new_t = self.terminal(s2, self.desired_goal()) or t
        self.trajectory.append((self.current_state, action, new_r, new_s2,
                                new_t))
        self.current_state = new_s2
        return new_s2, new_r, new_t, {'base_reward': r}

    def reset(self):
        new_state = State(self.env.reset(), self.desired_goal())
        self.trajectory = []
        self.current_state = new_state
        return new_state

    def recompute_trajectory(self):
        if not self.trajectory:
            return
        (_, _, _, sp_final, _) = self.trajectory[-1]
        achieved_goal = self.goal_from_obs_part(sp_final.obs)
        for (s, a, r, sp, t) in self.trajectory:
            new_s = s.obs, achieved_goal
            new_sp = sp.obs, achieved_goal
            new_r = self.reward(sp.obs, achieved_goal)
            new_t = self.terminal(sp.obs, achieved_goal) or t
            yield new_s, a, new_r, new_sp, new_t
            if new_t:
                break


class MountaincarGoalWrapper(GoalWrapper):
    """
    new obs is [pos, vel, goal_pos]
    """

    def goal_from_obs_part(self, obs_part):
        return np.array([obs_part[0]])

    def reward(self, obs_part, goal):
        return 100 if obs_part[0] >= goal[0] else 0

    def terminal(self, obs_part, goal):
        return obs_part[0] >= goal[0]

    def desired_goal(self):
        return np.array([0.45])


class PickAndPlaceGoalWrapper(GoalWrapper):
    def __init__(self, env):
        assert isinstance(env, PickAndPlaceEnv)
        super().__init__(env)

    def goal_from_obs_part(self, obs_part):
        return self.env.gripper_pos(obs_part[0]), self.env.block_pos()

    def reward(self, obs_part, goal):
        return sum(self.env.compute_reward(goal, obs) for obs in obs_part)

    def terminal(self, obs_part, goal):
        return any(self.env.compute_terminal(goal, obs) for obs in obs_part)

    def desired_goal(self):
        return self.env.goal()

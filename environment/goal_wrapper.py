from abc import abstractmethod
from collections import namedtuple

import gym
import numpy as np
from gym.envs.robotics import FetchReachEnv
from gym.envs.robotics.fetch_env import goal_distance
from gym.spaces import Box

from environment.pick_and_place import Goal, PickAndPlaceEnv
from sac.utils import Step

State = namedtuple('State', 'obs goal')


class HindsightWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        vector_state = self.vectorize(self.reset())
        self.observation_space = Box(-1, 1, vector_state.shape)

    @abstractmethod
    def achieved_goal(self, obs):
        raise NotImplementedError

    @abstractmethod
    def reward(self, obs, goal):
        raise NotImplementedError

    @abstractmethod
    def terminal(self, obs, goal):
        raise NotImplementedError

    @abstractmethod
    def desired_goal(self):
        raise NotImplementedError

    @staticmethod
    def vectorize(state):
        return np.concatenate(state)

    def step(self, action):
        s2, r, t, info = self.env.step(action)
        new_s2 = State(obs=s2, goal=self.desired_goal())
        new_r = self.reward(s2, self.desired_goal())
        new_t = self.terminal(s2, self.desired_goal()) or t
        return new_s2, new_r, new_t, {'base_reward': r}

    def reset(self):
        return State(obs=self.env.reset(), goal=self.desired_goal())

    def recompute_trajectory(self, trajectory):
        if not trajectory:
            return ()
        achieved_goal = self.achieved_goal(trajectory[-1].s2.obs)
        for step in trajectory:
            new_s = State(obs=step.s1.obs, goal=achieved_goal)
            new_sp = State(obs=step.s2.obs, goal=achieved_goal)
            new_r = self.reward(obs=step.s2.obs, goal=achieved_goal)
            new_t = self.terminal(obs=step.s2.obs, goal=achieved_goal) or step.t
            yield Step(s1=new_s, a=step.a, r=new_r, s2=new_sp, t=new_t)
            if new_t:
                break


class MountaincarHindsightWrapper(HindsightWrapper):
    """
    new obs is [pos, vel, goal_pos]
    """

    def achieved_goal(self, obs):
        return np.array([obs[0]])

    def reward(self, obs, goal):
        return 100 if obs[0] >= goal[0] else 0

    def terminal(self, obs, goal):
        return obs[0] >= goal[0]

    def desired_goal(self):
        return np.array([0.45])


class PickAndPlaceHindsightWrapper(HindsightWrapper):
    def __init__(self, env):
        assert isinstance(env, PickAndPlaceEnv)
        super().__init__(env)

    def achieved_goal(self, history):
        last_obs, = history[-1]
        return Goal(
            gripper=self.env.gripper_pos(last_obs),
            block=self.env.block_pos(last_obs))

    def reward(self, obs, goal):
        return sum(self.env.reward(goal, obs) for obs in obs)

    def terminal(self, obs, goal):
        return any(self.env.terminal(goal, obs) for obs in obs)

    def desired_goal(self):
        return self.env.goal()

    @staticmethod
    def vectorize(state):
        state = State(*state)
        state_history = list(map(np.concatenate, state.obs))
        return np.concatenate(
            [np.concatenate(state_history),
             np.concatenate(state.goal)])

    def step(self, action):
        s2, r, t, info = super().step(action)
        if t:
            s2 = self.reset()
        return s2, r, t, info


ACHIEVED_GOAL = 'achieved_goal'


class FetchReachHindsightWrapper(HindsightWrapper):
    def __init__(self, env):
        assert isinstance(env, FetchReachEnv)
        super().__init__(env)

    def achieved_goal(self, obs):
        return obs[ACHIEVED_GOAL]

    def reward(self, obs, goal):
        return self.env.compute_reward(obs[ACHIEVED_GOAL], goal, {})

    def terminal(self, obs, goal):
        return goal_distance(obs[ACHIEVED_GOAL], goal) < self.env.distance_threshold

    def desired_goal(self):
        return self.env.goal.copy()

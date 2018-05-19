from abc import abstractmethod
from collections import namedtuple

import gym
import numpy as np
from gym.spaces import Box

from environment.pick_and_place import Goal, PickAndPlaceEnv
from sac.utils import Step

State = namedtuple('State', 'obs goal')


class HindsightWrapper(gym.Wrapper):
    def __init__(self, env, default_reward=0):
        self._default_reward = default_reward
        super().__init__(env)
        vector_state = self.vectorize_state(self.reset())
        self.observation_space = Box(-1, 1, vector_state.shape)

    @abstractmethod
    def achieved_goal(self, obs):
        raise NotImplementedError

    @abstractmethod
    def at_goal(self, obs, goal):
        raise NotImplementedError

    @abstractmethod
    def desired_goal(self):
        raise NotImplementedError

    @staticmethod
    def vectorize_state(state):
        return np.concatenate(state)

    def step(self, action):
        s2, r, t, info = self.env.step(action)
        new_s2 = State(obs=s2, goal=self.desired_goal())
        new_r = float(self.at_goal(s2, self.desired_goal()))
        new_t = self.at_goal(s2, self.desired_goal()) or t
        return new_s2, new_r, new_t, {'base_reward': r}

    def reset(self):
        return State(obs=self.env.reset(), goal=self.desired_goal())

    def recompute_trajectory(self, trajectory):
        if not trajectory:
            return ()
        achieved_goal = self.achieved_goal(trajectory[-1].s2.obs)
        for step in trajectory:
            new_t = self.at_goal(step.s2.obs, achieved_goal) or step.t
            r = 1 if self.at_goal(step.s2.obs, achieved_goal) else self._default_reward
            yield Step(s1=State(obs=step.s1.obs, goal=achieved_goal),
                       a=step.a,
                       r=r,
                       s2=State(obs=step.s2.obs, goal=achieved_goal),
                       t=new_t)
            if new_t:
                break


class MountaincarHindsightWrapper(HindsightWrapper):
    """
    new obs is [pos, vel, goal_pos]
    """

    def achieved_goal(self, obs):
        return np.array([obs[0]])

    def at_goal(self, obs, goal):
        return obs[0] >= goal[0]

    def desired_goal(self):
        return np.array([0.45])


class PickAndPlaceHindsightWrapper(HindsightWrapper):
    def __init__(self, env, default_reward):
        if isinstance(env, gym.Wrapper):
            assert isinstance(env.unwrapped, PickAndPlaceEnv)
            self.unwrapped_env = env.unwrapped
        else:
            assert isinstance(env, PickAndPlaceEnv)
            self.unwrapped_env = env
        super().__init__(env, default_reward)

    def achieved_goal(self, history):
        last_obs, = history[-1]
        return Goal(
            gripper=self.unwrapped_env.gripper_pos(last_obs),
            block=self.unwrapped_env.block_pos(last_obs))

    def at_goal(self, obs, goal):
        return any(self.unwrapped_env.compute_terminal(goal, o) for o in obs)

    def desired_goal(self):
        return self.unwrapped_env.goal()

    @staticmethod
    def vectorize_state(state):
        state = State(*state)
        state_history = list(map(np.concatenate, state.obs))
        return np.concatenate(
            [np.concatenate(state_history),
             np.concatenate(state.goal)])

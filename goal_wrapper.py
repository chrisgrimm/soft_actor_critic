from abc import abstractmethod
import numpy as np
from environment.pick_and_place import PickAndPlaceEnv
from gym.spaces import Box


class GoalWrapper:

    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        s1 = env.reset()
        new_s1 = self.obs_from_obs_part_and_goal(s1, self.final_goal())
        self.observation_space = Box(-1, 1, new_s1.shape)
        self.trajectory = []
        self.current_state = new_s1

    @abstractmethod
    def obs_part_to_goal(self, obs_part):
        pass

    @abstractmethod
    def reward(self, obs_part, goal):
        pass

    @abstractmethod
    def terminal(self, obs_part, goal):
        pass

    @abstractmethod
    def get_obs_part(self, obs):
        pass

    @abstractmethod
    def get_goal_part(self, obs):
        pass

    @abstractmethod
    def obs_from_obs_part_and_goal(self, obs_part, goal):
        pass

    @abstractmethod
    def final_goal(self):
        pass

    def step(self, action):
        s2, r, t, info = self.env.step(action)
        new_s2 = self.obs_from_obs_part_and_goal(s2, self.final_goal())
        new_r = self.reward(s2, self.final_goal())
        new_t = self.terminal(s2, self.final_goal()) or t
        if new_t:
            print('goal_wrapper reward:', new_r)
        self.trajectory.append((self.current_state, action, new_r, new_s2, new_t))

        self.current_state = new_s2
        return new_s2, new_r, new_t, {'base_reward': r}

    def reset(self):
        s1 = self.env.reset()
        new_s1 = self.obs_from_obs_part_and_goal(s1, self.final_goal())
        self.trajectory = []
        self.current_state = new_s1
        return new_s1

    def render(self):
        return self.env.render()

    def recompute_trajectory(self):
        if not self.trajectory:
            return
        (_, _, _, sp_final, _) = self.trajectory[-1]
        final_goal = self.obs_part_to_goal(self.get_obs_part(sp_final))
        for (s, a, r, sp, t) in self.trajectory:
            s_obs_part = self.get_obs_part(s)
            sp_obs_part = self.get_obs_part(sp)

            new_s = self.obs_from_obs_part_and_goal(s_obs_part, final_goal)
            new_sp = self.obs_from_obs_part_and_goal(sp_obs_part, final_goal)
            new_r = self.reward(sp_obs_part, final_goal)
            new_t = self.terminal(sp_obs_part, final_goal) or t
            yield new_s, a, new_r, new_sp, new_t
            if new_t:
                break


class MountaincarGoalWrapper(GoalWrapper):
    """
    new obs is [pos, vel, goal_pos]
    """

    def obs_part_to_goal(self, obs_part):
        return np.array([obs_part[0]])

    def reward(self, obs_part, goal):
        return 100 if obs_part[0] >= goal[0] else 0

    def terminal(self, obs_part, goal):
        return obs_part[0] >= goal[0]

    def get_obs_part(self, obs):
        return obs[:2]

    def get_goal_part(self, obs):
        return np.array([obs[2]])

    def obs_from_obs_part_and_goal(self, obs_part, goal):
        return np.concatenate([obs_part, goal], axis=0)

    def final_goal(self):
        return np.array([0.45])


class PickAndPlaceGoalWrapper(GoalWrapper):
    def __init__(self, env):
        assert isinstance(env, PickAndPlaceEnv)
        super().__init__(env)

    def obs_part_to_goal(self, obs_part):
        return self.env._obs_to_goal(obs_part[-1])

    def reward(self, obs_part, goal):
        return sum(self.env._compute_reward(goal, obs) for obs in obs_part)

    def terminal(self, obs_part, goal):
        return any(self.env._compute_terminal(goal, obs) for obs in obs_part)

    def get_obs_part(self, obs):
        goal, obs_history = self.env.destructure_mlp_input(obs)
        return obs_history

    def get_goal_part(self, obs):
        return self.env.obs_to_goal(obs)

    def obs_from_obs_part_and_goal(self, obs_part, goal):
        return self.env.mlp_input(goal, obs_part)

    def final_goal(self):
        return self.env._goal()

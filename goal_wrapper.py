from abc import abstractmethod
import numpy as np
from gym.spaces import Box

class GoalWrapper(object):

    def __init__(self, env, buffer, reward_scaling):
        self.env = env
        self.buffer = buffer
        self.action_space = self.env.action_space
        s1 = env.reset()
        new_s1 = self.obs_from_obs_part_and_goal(s1, self.final_goal())
        #self.observation_space = Box(-1, 1, new_s1.shape)
        self.current_trajectory = []
        self.current_state = new_s1
        self.reward_scaling = reward_scaling

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

    def step(self, action, action_converter):
        s2, r, t, info = self.env.step(action_converter(action))
        new_s2 = self.obs_from_obs_part_and_goal(s2, self.final_goal())
        new_r = self.reward(s2, self.final_goal())
        new_t = self.terminal(s2, self.final_goal()) or t
        #new_t = t
        self.current_trajectory.append((self.current_state, action, new_r, new_s2, new_t))

        self.current_state = new_s2
        return new_s2, new_r, new_t, {'base_reward': r}


    def reset_called(self):
        pass

    def reset(self):
        self.feed_new_trajectory_to_buffer(self.current_trajectory)
        s1 = self.env.reset()
        self.reset_called()
        new_s1 = self.obs_from_obs_part_and_goal(s1, self.final_goal())
        self.current_trajectory = []
        self.current_state = new_s1
        return new_s1

    def render(self):
        return self.env.render()



    def recompute_trajectory(self, trajectory):
        if not trajectory:
            return
        (_, _, _, sp_final, _) = trajectory[-1]
        final_goal = self.obs_part_to_goal(self.get_obs_part(sp_final))
        for (s, a, r, sp, t) in trajectory:
            s_obs_part = self.get_obs_part(s)
            sp_obs_part = self.get_obs_part(sp)

            new_s = self.obs_from_obs_part_and_goal(s_obs_part, final_goal)
            new_sp = self.obs_from_obs_part_and_goal(sp_obs_part, final_goal)
            new_r = self.reward(sp_obs_part, final_goal)
            new_t = self.terminal(sp_obs_part, final_goal) or t
            yield new_s, a, new_r, new_sp, new_t
            if new_t == True:
                break

    def feed_new_trajectory_to_buffer(self, trajectory):
        for (s, a, r, sp, t) in self.recompute_trajectory(trajectory):
            self.buffer.append(s, a, r / self.reward_scaling, sp ,t)


class MountaincarGoalWrapper(GoalWrapper):
    '''
    new obs is [pos, vel, goal_pos]
    '''
    def obs_part_to_goal(self, obs_part):
        return np.array([obs_part[0]])

    def reward(self, obs_part, goal):
        return 100 if obs_part[0] >= goal[0] else 0
        #dist = np.abs(obs_part[0] - goal[0])
        #return 100 if dist < 0.03 else 0

    def terminal(self, obs_part, goal):
        #dist = np.abs(obs_part[0] - goal[0])
        return (obs_part[0] >= goal[0])
        #return (dist < 0.03)

    def get_obs_part(self, obs):
        return obs[:2]

    def get_goal_part(self, obs):
        return np.array([obs[2]])

    def obs_from_obs_part_and_goal(self, obs_part, goal):
        return np.concatenate([obs_part, goal], axis=0)

    def final_goal(self):
        return np.array([0.45])

import numpy as np
import os
import tensorflow as tf
from gym.spaces import Box, Discrete
from abc import abstractmethod
from networks.policy_mixins import CNN_Power2_Policy, MLPPolicy, GaussianPolicy
from networks.value_function_mixins import CNN_Power2_ValueFunc, MLPValueFunc
from networks.network_interface import AbstractSoftActorCritic
from column_game import ColumnGame
from indep_control2.vae_network import VAE_Network
import re

def build_action_converter(env):
    def converter(a):
        if type(env.action_space) is Discrete:
                return np.argmax(a)
        else:
            a = np.tanh(a)
            h, l = env.action_space.high, env.action_space.low
            return ((a + 1) / 2) * (h - l) + l
    return converter

def build_column_agent(env, name, sess=None):
    if env.visual:
        class Agent(GaussianPolicy, CNN_Power2_Policy, CNN_Power2_ValueFunc, AbstractSoftActorCritic):
            def __init__(self, s_shape, a_shape):
                super(Agent, self).__init__(s_shape, a_shape, global_name=name, sess=sess)
    else:
        class Agent(GaussianPolicy, MLPPolicy, MLPValueFunc, AbstractSoftActorCritic):
            def __init__(self, s_shape, a_shape):
                super(Agent, self).__init__(s_shape, a_shape, global_name=name, sess=sess)

    return Agent(env.observation_space.shape, env.action_space.shape)


class HighLevelColumnEnvironment():

    def __init__(self, max_l0_steps=100, perfect_agents=False, hindsight=False, buffer=None):
        self.num_factors = 20
        nn = VAE_Network(self.num_factors, 10*10, mode='image')
        nn.restore('./indep_control2/vae_network.ckpt')
        self.env = ColumnGame(nn, force_max=0.3, reward_per_goal=10.0, indices=None, visual=False, max_episode_steps=100)

        self.perfect_agents = perfect_agents
        # factors that are used.
        self.index_to_factor = [4, 5, 7, 8, 9, 10, 11, 16]
        self.factor_to_column = {4:1, 5:7, 7:6, 8:3, 9:4, 10:2, 11:0, 16:5}

        if not perfect_agents:
            self.sess = None
            option_dir = './factor_agents'
            option_names = [f'f{x}_random' for x in self.index_to_factor]
            scope_names = [f'SAC{x}' for x in self.index_to_factor]
            option_paths = [os.path.join(option_dir, name, 'weights/sac.ckpt') for name in option_names]
            self.agents = [self.load_agents(path, scope) for path, scope in zip(option_paths, scope_names)]
            self.max_l0_steps = max_l0_steps
            self.l0_action_converter = build_action_converter(self.env)

        # high-level objective parameters.
        self.goal_threshold = self.env.goal_threshold

        self.hindsight = hindsight
        self.current_episode = []
        self.buffer = buffer
        self.min_dist = np.inf
        if self.hindsight:
            assert buffer is not None




    def load_agents(self, option_path, global_scope):
        print(f'loading {global_scope}...')
        agent = build_column_agent(self.env, global_scope, self.sess)
        if self.sess is None:
            self.sess = agent.sess
        agent.restore(option_path)
        return agent

    def insert_hindsight_trajectory(self, idx):
        if idx != -1:
            idx_pos = idx
            idx_range = slice(0, idx)
        else:
            idx_pos = idx_range = idx


        if len(self.current_episode) == 0:
            return
        new_trajectory = []
        last_s1, last_a, last_r, last_s2, last_t = self.current_episode[idx_pos]
        goal = last_s2[:20]
        for (s1, a, r, s2, t) in self.current_episode[idx_range]:
            new_s1 = np.copy(np.concatenate([s1[:20], goal], axis=0))
            new_a = np.copy(a)
            new_s2 = np.copy(np.concatenate([s2[:20], goal], axis=0))
            new_at_goal = self.env.at_goal(new_s2[:20], goal)
            new_r = self.get_reward(new_s2, new_s1, goal)
            new_t = new_at_goal
            #new_trajectory.append((new_s1, new_a, new_r, new_s2, new_t))
            self.buffer.append(new_s1, new_a, new_r, new_s2, new_t)
            #print('boop')






    def step(self, action, action_converter):
        converted_action = action_converter(action)
        (agent_index, parameter) = converted_action
        factor_num = self.index_to_factor[agent_index]
        l0_goal = np.zeros(shape=[self.num_factors], dtype=np.float32)
        l0_goal[factor_num] = parameter

        # perform action using the agent.
        #l1_reward = self.env.reward_no_goal
        l1_terminal = False
        terminal = False
        # compute old dist.
        #old_column_positions = np.copy(self.env.column_positions)

        if not self.perfect_agents:
            selected_agent = self.agents[agent_index]
            obs = self.env.get_observation()
            old_obs = np.copy(obs)

            for i in range(self.max_l0_steps):
                obs = self.env.get_observation()
                obs_goal = np.concatenate([obs[:20], l0_goal], axis=0)
                l0_action = selected_agent.get_actions([obs_goal])[0]
                l0_action = self.l0_action_converter(l0_action)
                obs, reward, terminal, info = self.env.step(l0_action)

                # kill the option if the l1_goal is reached.
                at_goal = self.env.at_goal(obs[:20], self.env.goal)
                if at_goal:
                    l1_terminal = True
                    break
                # kill the option if the episode ends.
                if terminal:
                    break

                # kill the option if the option-goal is reached.
                if self.env.at_goal(obs[:20], l0_goal, indices=[factor_num]):
                    break

        else:
            old_obs = self.env.get_observation()
            old_column_pos = np.copy(self.env.column_positions)
            desired_column = self.factor_to_column[factor_num]
            self.env.column_positions[desired_column] = parameter
            #print(parameter)

            obs = self.env.get_observation()
            self.env.episode_step += 1
            #at_goal = self.env.at_goal(obs[:20], self.env.goal)
            at_goal = self.env.at_goal(self.env.column_positions, self.env.goal_column_heights, indices=range(8))
            if at_goal:
                l1_terminal = True

            if self.env.episode_step >= self.env.max_episode_steps:
                terminal = True

        obs = self.env.get_observation()
        column_pos = np.copy(self.env.column_positions)
        # compute new distance
        #new_dist = self.dist_to_goal(self.env.column_positions, self.goal)
        #l1_reward = 20 * (old_dist - new_dist)
        #dist_to_goal_total = np.max(np.abs(obs[:20] - self.env.goal))
        dist_to_goal_total = np.mean(np.abs(self.env.column_positions - self.env.goal_column_heights))
        self.min_dist = np.minimum(dist_to_goal_total, self.min_dist)
        #print(np.all(obs[20:] == self.env.goal), dist_to_goal_total)
        l1_reward = self.get_reward(column_pos, old_column_pos, np.copy(self.env.goal_column_heights))
        terminal = terminal or l1_terminal


        self.current_episode.append((np.copy(old_obs), np.copy(action), l1_reward, np.copy(obs), terminal))

        return obs, l1_reward, terminal, {}

    def dist_to_goal(self, column_positions, goal):
        return np.mean(np.abs(column_positions - goal))

    def get_reward(self, obs, old_obs, goal):
        #l1_reward = self.env.reward_per_goal if self.env.at_goal(obs[:20], goal) else self.env.reward_no_goal

        new_dist = self.dist_to_goal(obs[:20], goal)
        old_dist = self.dist_to_goal(old_obs[:20], goal)
        l1_reward = 20*(old_dist - new_dist)
        #l1_reward = -20*new_dist

        return l1_reward


    def reset(self):
        #self.insert_hindsight_trajectory(-1)
        #for i in range(4):
        #    if len(self.current_episode) == 0:
        #        break
        #     self.insert_hindsight_trajectory(np.random.randint(0, len(self.current_episode)))
        print('min_dist', self.min_dist)
        self.min_dist = np.inf
        self.current_episode = []
        return self.env.reset()

    def render(self):
        return self.env.render()


def build_action_converter2(env):
    column_to_factor = {v: k for k, v in env.factor_to_column.items()}
    factor_to_index = {f: i for i, f in enumerate(env.index_to_factor)}
    def converter(action):
        (column_number, value) = action
        factor = column_to_factor[column_number]
        index = factor_to_index[factor]
        return (index, value)
    return converter

if __name__ == '__main__':
    env = HighLevelColumnEnvironment(perfect_agents=True)
    s = env.reset()
    t = False
    print(f'obs: {s[:20]}\ncolumns: {env.env.column_positions}\ngoal: {env.env.goal_column_heights}')
    converter = build_action_converter2(env)
    while True:
        text_input = input('action:')
        if text_input == "-1":
            s = env.reset()
            #env.render()
            continue
        (action_num, goal) = re.match(r'^(\d+) ([\-\d\.]+)$', text_input).groups()
        action_num = int(action_num)
        goal = float(goal)
        s, r, t, info = env.step((action_num, goal), converter)
        print(f'obs: {s[:20]}\ncolumns: {env.env.column_positions}\ngoal: {env.env.goal_column_heights}')
        if t:
            print(f'terminal! reward {r}')
            s = env.reset()
        env.render()









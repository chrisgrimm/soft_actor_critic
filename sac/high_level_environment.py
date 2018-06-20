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

    def __init__(self, max_l0_steps=100, perfect_agents=False):
        self.num_factors = 20
        nn = VAE_Network(self.num_factors, 10*10, mode='image')
        nn.restore('./indep_control2/vae_network.ckpt')
        self.env = ColumnGame(nn, force_max=0.3, reward_per_goal=10.0, indices=None, visual=False, max_episode_steps=100)
        self.perfect_agents = perfect_agents
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
        self.goal = np.ones(shape=[8]) * 0.5
        self.goal_threshold = self.env.goal_threshold


    def load_agents(self, option_path, global_scope):
        print(f'loading {global_scope}...')
        agent = build_column_agent(self.env, global_scope, self.sess)
        if self.sess is None:
            self.sess = agent.sess
        agent.restore(option_path)
        return agent


    def step(self, action):
        (agent_index, parameter) = action
        factor_num = self.index_to_factor[agent_index]
        goal = np.zeros(shape=[self.num_factors], dtype=np.float32)
        goal[factor_num] = parameter

        # perform action using the agent.
        #l1_reward = self.env.reward_no_goal
        l1_terminal = False
        terminal = False
        if not self.perfect_agents:
            selected_agent = self.agents[agent_index]

            obs = self.env.get_observation()

            for i in range(self.max_l0_steps):
                obs = self.env.get_observation()
                obs_goal = np.concatenate([obs[:20], goal], axis=0)
                action = selected_agent.get_actions([obs_goal])[0]
                action = self.l0_action_converter(action)
                obs, reward, terminal, info = self.env.step(action)

                # kill the option if the l1_goal is reached.
                at_goal, dist_to_goal = self.at_goal()
                if at_goal:
                    l1_terminal = True
                    l1_reward = self.env.reward_per_goal
                    break

                if terminal:
                    break

                if self.env.at_goal(obs[:20], goal, indices=[factor_num]):
                    break
        else:
            old_column_positions = np.copy(self.env.column_positions)
            old_dist = self.dist_to_goal(old_column_positions, self.goal)

            desired_column = self.factor_to_column[factor_num]
            self.env.column_positions[desired_column] = parameter
            new_dist = self.dist_to_goal(self.env.column_positions, self.goal)
            l1_reward = 20*(old_dist - new_dist)
            #print(parameter)
            obs = self.env.get_observation()
            self.env.episode_step += 1
            at_goal, dist_to_goal = self.at_goal()

            #l1_reward = -dist_to_goal
            if at_goal:
                l1_terminal = True
                #l1_reward = -10*dist_to_goal
                #l1_reward = self.env.reward_per_goal
            if self.env.episode_step >= self.env.max_episode_steps:
                terminal = True
                #l1_reward = -10*dist_to_goal

        return obs, l1_reward, terminal or l1_terminal, {}

    def dist_to_goal(self, column_positions, goal):
        return np.mean(np.abs(column_positions - goal))

    def at_goal(self):
        dist_to_goal = self.dist_to_goal(self.env.column_positions, self.goal)
        return dist_to_goal < self.goal_threshold, dist_to_goal


    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()



if __name__ == '__main__':
    env = HighLevelColumnEnvironment(perfect_agents=True)
    s = env.reset()
    t = False
    while True:
        text_input = input('action:')
        if text_input == "-1":
            s = env.reset()
            env.render()
            continue
        (action_num, goal) = re.match(r'^(\d+) ([\-\d\.]+)$', text_input).groups()
        action_num = int(action_num)
        goal = float(goal)
        s, r, t, info = env.step((action_num, goal))
        if t:
            s = env.reset()
        env.render()









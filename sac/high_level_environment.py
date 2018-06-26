import numpy as np
import os
import tensorflow as tf
from gym.spaces import Box, Discrete
from abc import abstractmethod
from networks.policy_mixins import CNN_Power2_Policy, MLPPolicy, GaussianPolicy
from networks.value_function_mixins import CNN_Power2_ValueFunc, MLPValueFunc
from networks.network_interface import AbstractSoftActorCritic
from column_game import ColumnGame, make_n_columns
from indep_control2.vae_network import VAE_Network
import re
import cv2

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

    def __init__(self, max_l0_steps=100, perfect_agents=False, buffer=None):
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
        #self.goal, self.goal_encoded = self.new_goal()
        self.goal = np.ones(shape=[8])*0.5
        self.goal_encoded = self.env.nn.encode_deterministic([self.goal])[0]
        self.goal_threshold = self.env.goal_threshold
        self.buffer = buffer
        self.current_episode = []


    def new_goal(self):
        heights = np.random.uniform(0, 1, size=[8])
        encoded_goal = self.env.nn.encode_deterministic([make_n_columns(heights, spacing=self.env.spacing, size=self.env.image_size)])[0]
        return heights, encoded_goal

    def load_agents(self, option_path, global_scope):
        print(f'loading {global_scope}...')
        agent = build_column_agent(self.env, global_scope, self.sess)
        if self.sess is None:
            self.sess = agent.sess
        agent.restore(option_path)
        return agent

    def store_hindsight_experience(self):
        if len(self.current_episode) == 0:
            return
        new_goal = np.copy(self.current_episode[-1][3][:8])
        for (s, a, r, sp, t) in self.current_episode:
            new_s = np.copy(np.concatenate([s[:8], new_goal], axis=0))
            new_sp = np.copy(np.concatenate([sp[:8], new_goal], axis=0))
            new_a = np.copy(a)
            new_r = self.compute_reward(new_sp, new_s, new_goal)
            new_t = self.compute_terminal(new_sp, new_goal)
            self.buffer.append(new_s, new_a, new_r, new_sp, new_t)




    def step(self, raw_action, action_converter):
        action = action_converter(raw_action)
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
            # get the old observation
            old_obs = self.get_observation()
            old_column_positions = np.copy(self.env.column_positions)

            # modify the column positions
            desired_column = self.factor_to_column[factor_num]
            self.env.column_positions[desired_column] = parameter


            # get the new observation
            obs = self.get_observation()
            column_positions = np.copy(self.env.column_positions)

            # compute the reward and terminality
            l1_reward = self.compute_reward(column_positions, old_column_positions, self.goal)
            at_goal, goal_dist = self.compute_terminal(column_positions, self.goal, return_dist=True)

            self.env.episode_step += 1

            if at_goal:
                l1_terminal = True
            if self.env.episode_step >= self.env.max_episode_steps:
                terminal = True
                #l1_reward = -10*dist_to_goal

        if terminal or l1_terminal:
            print(f'goal_dist: {goal_dist}')

        self.current_episode.append((np.concatenate([old_column_positions, self.goal], axis=0), raw_action, l1_reward, np.concatenate([column_positions, self.goal], axis=0), terminal or l1_terminal))
        return np.concatenate([column_positions, self.goal], axis=0), l1_reward, terminal or l1_terminal, {}

    def get_observation(self):
        return np.concatenate([self.env.get_observation()[:20], self.goal_encoded], axis=0)

    def compute_reward(self, obs, old_obs, goal):
        old_dist = self.dist_to_goal(old_obs[:8], goal)
        new_dist = self.dist_to_goal(obs[:8], goal)
        return 20*(old_dist - new_dist)

    def compute_terminal(self, obs, goal, return_dist=False):
        return self.env.at_goal(obs[:8], goal, return_dist=return_dist)


    def dist_to_goal(self, column_positions, goal):
        return np.mean(np.abs(column_positions - goal))

    def at_goal(self):
        dist_to_goal = self.dist_to_goal(self.env.column_positions, self.goal)
        return dist_to_goal < self.goal_threshold, dist_to_goal


    def reset(self):
        #self.store_hindsight_experience()
        #self.goal, self.goal_encoded = self.new_goal()
        self.current_episode = []
        self.env.reset()
        return np.concatenate([np.copy(self.env.column_positions), self.goal], axis=0)

    def render(self):
        return self.env.render()


class DummyHighLevelEnv(object):

    def __init__(self, sparse_reward=False, goal_reward=10, no_goal_penalty=-0.1, goal_threshold=0.1, buffer=None):
        # environment hyperparameters
        self.sparse_reward = sparse_reward
        self.goal_reward = goal_reward
        self.no_goal_penalty = no_goal_penalty
        self.goal_threshold = goal_threshold


        self.column_position = self.new_column_position()
        self.goal = self.new_column_position()
        self.num_steps = 0
        self.max_steps = 100

        # hindsight stuff
        self.current_trajectory = []
        self.buffer = buffer

    def add_hindsight_experience(self):
        if len(self.current_trajectory) == 0:
            return
        _, _, _, last_sp, _ = self.current_trajectory[-1]
        goal = np.copy(last_sp[:8])
        for s, a, r, sp, t in self.current_trajectory:
            new_s = np.copy(np.concatenate([s[:8], goal], axis=0))
            new_a = np.copy(a)
            new_sp = np.copy(np.concatenate([sp[:8], goal], axis=0))
            new_r = self.get_reward(new_sp)
            new_t = self.get_terminal(new_sp)
            self.buffer.append(new_s, new_a, new_r, new_sp, new_t)

    def step(self, raw_action, action_converter):
        action = action_converter(raw_action)
        (column_index, parameter) = action

        old_obs = self.get_observation()

        self.column_position[column_index] = parameter
        self.num_steps += 1

        obs = self.get_observation()

        reward = self.get_reward(obs)

        terminal = self.get_terminal(obs) or self.num_steps >= self.max_steps
        if terminal:
            print('dist_to_goal', self.dist_to_goal(obs[:8], obs[8:]))

        if self.buffer is not None:
            self.current_trajectory.append((old_obs, raw_action, reward, obs, terminal))

        return obs, reward, terminal, {}

    def new_column_position(self):
        return np.random.uniform(0, 1, size=[8])

    def get_observation(self, goal=None):
        goal = np.copy(self.goal) if goal is None else goal
        return np.concatenate([np.copy(self.column_position), goal], axis=0)

    def get_reward(self, obs, goal=None):
        goal = obs[8:] if goal is None else goal
        obs_part = obs[:8]

        if self.sparse_reward:
            return self.goal_reward if self.get_terminal(obs, goal) else self.no_goal_penalty
        else:
            new_dist = self.dist_to_goal(obs_part, goal)
            return -20*new_dist

    def get_terminal(self, obs, goal=None):
        goal = obs[8:] if goal is None else goal
        obs_part = obs[:8]
        return self.dist_to_goal(obs_part, goal) < 0.1


    def dist_to_goal(self, obs_part, goal):
        return np.mean(np.abs(obs_part - goal))

    def reset(self):
        if self.buffer is not None:
            self.add_hindsight_experience()
            self.current_trajectory = []
        self.column_position = self.new_column_position()
        self.goal = self.new_column_position()
        self.num_steps = 0
        return self.get_observation()

    def render(self):
        cv2.imshow('game', 255 * make_n_columns(self.column_position, spacing=2, size=128))
        cv2.waitKey(1)




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









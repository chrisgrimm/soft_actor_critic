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
        new_goal = np.copy(self.current_episode[-1][3][:])
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

    def __init__(self, sparse_reward=False, goal_reward=10, no_goal_penalty=-0.1, goal_threshold=0.1, buffer=None,
                 distance_mode='mean', hindsight_strategy='final', num_columns=8, use_encoding=False,
                 centered_actions=False, accept_discrete_and_gaussian=False, use_l0_agents=False,
                 single_column=-1, use_environment=False):
        # environment hyperparameters
        self.sparse_reward = sparse_reward
        self.goal_reward = goal_reward
        self.no_goal_penalty = no_goal_penalty
        self.goal_threshold = goal_threshold
        self.obs_size = num_columns
        self.spacing = 2
        self.image_size = 128
        self.num_columns = num_columns
        self.possible_distance_modes = ['mean', 'sum', 'max']
        self.possible_hindsight_strategies = ['final']
        self.centered_actions = centered_actions
        self.accept_discrete_and_gaussian = accept_discrete_and_gaussian
        self.single_column = single_column
        self.singled_obs_index = single_column
        self.centered_action_scaling = 1.0
        self.use_environment = use_environment

        # set up centered action mode by default if single-column mode is on.
        if self.single_column != -1:
            self.centered_action_scaling = 0.1
            self.centered_actions = True

        if self.single_column != -1:
            # max is the only distance mode that is invariant to the number of columns, ie. when single column mode is on,
            # max distance still makes sense to use.
            assert distance_mode == 'max'

        try:
            assert distance_mode in self.possible_distance_modes
        except AssertionError:
            raise Exception(f'Distance mode must be in list: {self.possible_distance_modes}')
        try:
            future_match = re.match(r'^future(\d+)$', hindsight_strategy)
            if future_match:
                self.future_strategy_k = int(future_match.groups()[0])
                self.hindsight_strategy = 'future'
            else:
                assert hindsight_strategy in self.possible_hindsight_strategies
                self.hindsight_strategy = hindsight_strategy

        except AssertionError:
            raise Exception(f'Hindsight strategy must be in list: {self.possible_hindsight_strategies}')
        self.distance_mode = distance_mode



        self.num_steps = 0
        self.max_steps = 100

        # hindsight stuff
        self.current_trajectory = []
        self.buffer = buffer

        # encoding stuff
        self.use_encoding = use_encoding
        self.use_l0_agents = use_l0_agents
        self.num_factors = 20

        self.index_to_factor = [4, 5, 7, 8, 9, 10, 11, 16]  # maps the agent index to the factor it controls
        self.factor_to_column = {4: 1, 5: 7, 7: 6, 8: 3, 9: 4, 10: 2, 11: 0, 16: 5}  # maps the factor to the column
        self.column_to_factor = {column : factor for factor, column in self.factor_to_column.items()}
        self.factor_to_index = {factor : index for index, factor in enumerate(self.index_to_factor)}

        if self.use_environment:
            self.env = ColumnGame(self.nn, force_max=0.3, reward_per_goal=10.0, visual=False, max_episode_steps=self.max_steps)

        if self.use_encoding:
            assert self.num_columns == 8
            self.obs_size = 20
            self.nn = VAE_Network(self.num_factors, 10*10, mode='image')
            self.nn.restore('./indep_control2/vae_network.ckpt')
            # make a column game for each agent.
            self.factor_envs = [ColumnGame(self.nn, force_max=0.3, reward_per_goal=10.0, indices=[factor], visual=False, max_episode_steps=self.max_steps)
                                for factor in self.index_to_factor]
            if self.single_column != -1:
                self.singled_obs_index = self.column_to_factor[self.single_column]

        if self.use_l0_agents:
            sess = None
            option_dir = './factor_agents'
            option_names = [f'f{x}_random' for x in self.index_to_factor]
            scope_names = [f'SAC{x}' for x in self.index_to_factor]
            option_paths = [os.path.join(option_dir, name, 'weights/sac.ckpt') for name in option_names]
            self.agents = [self.load_agents(path, scope) for path, scope in zip(option_paths, scope_names)]
            self.l0_action_converter = build_action_converter(self.factor_envs[0])




        self.observation_space = Box(-3, 3, shape=[2*self.obs_size], dtype=np.float32)
        if self.centered_actions:
            if self.accept_discrete_and_gaussian:
                self.action_space = Box(-1, 1, shape=[self.num_columns + 1], dtype=np.float32)
            else:
                self.action_space = Box(-1, 1, shape=[2], dtype=np.float32)
        else:
            if self.accept_discrete_and_gaussian:
                self.action_space = Box(0, 1, shape=[self.num_columns + 1], dtype=np.float32)
            else:
                self.action_space = Box(0, 1, shape=[2], dtype=np.float32)


        # initialize the environment
        self.set_column_position(self.new_column_position())
        self.goal = self.new_goal()

    def set_column_position(self, value):
        if self.use_environment:
            self.env.column_positions = value
        else:
            self.column_position = value

    def get_column_position(self):
        if self.use_environment:
            return np.copy(self.env.column_positions)
        else:
            return np.copy(self.column_position)



    def load_agents(self, option_path, global_scope):
        print(f'loading {global_scope}...')
        agent = build_column_agent(self.env, global_scope, self.sess)
        if self.sess is None:
            self.sess = agent.sess
        agent.restore(option_path)
        return agent

    def do_final_strategy(self):
        self.add_hindsight_experience(-1)

    def do_future_strategy(self, k):
        if len(self.current_trajectory) == 0:
            return
        for i in range(k):
            index = np.random.randint(0, len(self.current_trajectory))
            self.add_hindsight_experience(index)


    def add_hindsight_experience(self, index, use_value_estimates=False):
        if len(self.current_trajectory) == 0:
            return
        _, _, _, last_sp, _ = self.current_trajectory[index]
        #goal = np.copy(last_sp[:self.obs_size])
        goal = self.make_goal_from_obs(last_sp)
        for s, a, r, sp, t in self.current_trajectory[:index] + [self.current_trajectory[index]]:
            new_s = np.copy(np.concatenate([s[:self.obs_size], goal], axis=0))
            new_a = np.copy(a)
            new_sp = np.copy(np.concatenate([sp[:self.obs_size], goal], axis=0))
            new_r = self.get_reward(new_sp)
            new_t = self.get_terminal(new_sp)
            self.buffer.append(new_s, new_a, new_r, new_sp, new_t)

    def make_goal_from_obs(self, obs):
        if self.single_column == -1:
            return np.copy(obs[:self.obs_size])
        else:
            goal = np.zeros_like(obs[:self.obs_size])
            goal[self.singled_obs_index] = obs[self.singled_obs_index]
            return goal

    def step(self, raw_action):
        action = self.action_converter(raw_action)
        (column_index, parameter) = action

        old_obs = self.get_observation()

        if self.use_l0_agents:
            self.perform_action_l0_agents(column_index, parameter)
        else:
            self.perform_action(column_index, parameter)

        self.num_steps += 1

        obs = self.get_observation()

        reward = self.get_reward(obs)

        terminal = self.get_terminal(obs) or self.num_steps >= self.max_steps
        if terminal:
            print('dist_to_goal', self.dist_to_goal(obs[:self.obs_size], obs[self.obs_size:]))

        if self.buffer is not None:
            self.current_trajectory.append((old_obs, raw_action, reward, obs, terminal))

        return obs, reward, terminal, {}

    def perform_action(self, column_index, parameter):
        new_column_position = self.get_column_position()
        if not self.centered_actions:
            new_column_position[column_index] = parameter
            self.set_column_position(new_column_position)
        else:
            new_column_position[column_index] = np.clip(new_column_position[column_index] + self.centered_action_scaling * parameter, 0, 1)
            self.set_column_position(new_column_position)

    # we have an environment that sets the column value.
    # we want to take

    def perform_action_l0_agents(self, column_index, parameter):
        column_positions = self.get_column_position()
        self.env.reset()
        agent = self.agents[column_index]
        # how do I manage the low-level domain's state?
        # the low level domain has a way of updating the internal state when the agent takes an action.
        # expose this in the environment, and call this function on the state-representation in the high-level env.
        goal = np.zeros(shape=[20])
        # TODO this might not be right.
        goal[self.index_to_factor[column_index]] = parameter
        for i in range(self.max_steps):
            # pass proper observation into agent.
            obs = self.get_observation(goal=goal)
            action = agent.get_actions([obs])[0]
            action = self.l0_action_converter(action)
            self.column_position = self.env.apply_action_to_state(action, self.column_position)
            obs = self.get_observation(goal=goal)
            # stop if the option tells us to stop.
            if self.env.at_goal(obs[:20], goal):
                break
            # stop if the environmnet enters a terminal state.
            if self.get_terminal(obs, self.goal):
                break





    def action_converter(self, raw_action):
        if self.accept_discrete_and_gaussian:
            a_cat = np.argmax(raw_action[:self.num_columns])
            a_gauss = raw_action[self.num_columns]
        else:
            a_cat, a_gauss = raw_action[0], raw_action[1]
            a_cat = np.tanh(a_cat)
            a_cat = int((a_cat + 1) / 2.0 * self.num_columns)
            # handles stupid case when
            if a_cat == self.num_columns:
                a_cat = self.num_columns - 1
        # otherwise a_cat is just itself.
        a_gauss = np.tanh(a_gauss)
        # h, l = 2.5, -2.5
        h, l = 1.0, 0.0
        if not self.centered_actions:
            a_gauss = ((a_gauss + 1) / 2) * (h - l) + l
        return (a_cat, a_gauss)


    def new_column_position(self):
        return np.random.uniform(0, 1, size=[self.num_columns])

    def new_goal(self):
        if self.use_encoding:
            column_image = make_n_columns(self.new_column_position(), spacing=self.spacing, size=self.image_size)
            encoding = self.nn.encode_deterministic([column_image])[0]
            # if single-column is enabled, zero out
            if self.single_column != -1:
                new_encoding = np.zeros_like(encoding)
                factor = self.column_to_factor[self.single_column]
                new_encoding[factor] = encoding[factor]
                encoding = new_encoding
            return encoding
        else:
            column_position = self.new_column_position()
            if self.single_column != -1:
                new_column_position = np.zeros_like(column_position)
                new_column_position[self.single_column] = column_position[self.single_column]
                column_position = new_column_position
            return column_position

    def get_observation(self, goal=None):
        goal = np.copy(self.goal) if goal is None else goal
        column_position = self.get_column_position()
        if self.use_encoding:
            column_image = make_n_columns(column_position, spacing=self.spacing, size=self.image_size)
            encoding = self.nn.encode_deterministic([column_image])[0]
            return np.concatenate([encoding, goal], axis=0)
        else:
            return np.concatenate([column_position, goal], axis=0)

    def get_reward(self, obs, goal=None):
        goal = obs[self.obs_size:] if goal is None else goal
        obs_part = obs[:self.obs_size]
        if self.sparse_reward:
            goal_reward = self.goal_reward if self.get_terminal(obs, goal) else self.no_goal_penalty
            if self.single_column != -1:
                movement_penalty = self.compute_unnecessary_movement_penalty(obs_part, self.starting_position, [self.singled_obs_index])
            else:
                movement_penalty = 0
            return goal_reward - movement_penalty
        else:
            new_dist = self.dist_to_goal(obs_part, goal)
            if self.single_column != -1:
                movement_penalty = self.compute_unnecessary_movement_penalty(obs_part, self.starting_position, [self.singled_obs_index])
            else:
                movement_penalty = 0
            return -20*(new_dist - movement_penalty)

    def compute_unnecessary_movement_penalty(self, vector, starting_vector, indices=None):
        # penalize changes to the features on the unselected indices.
        vector = np.copy(vector)
        vector[indices] = 0
        starting_vector = np.copy(starting_vector)
        starting_vector[indices] = 0
        return np.sum(np.abs(vector - starting_vector))

    def get_terminal(self, obs, goal=None):
        goal = obs[self.obs_size:] if goal is None else goal
        obs_part = obs[:self.obs_size]
        if self.single_column != -1:
            new_obs_part = np.zeros_like(obs_part)
            new_obs_part[self.singled_obs_index] = obs_part[self.singled_obs_index]
            obs_part = new_obs_part
        return self.dist_to_goal(obs_part, goal) < self.goal_threshold


    def dist_to_goal(self, obs_part, goal):
        if self.distance_mode == 'sum':
            return np.sum(np.abs(obs_part - goal))
        elif self.distance_mode == 'mean':
            return np.mean(np.abs(obs_part - goal))
        elif self.distance_mode == 'max':
            return np.max(np.abs(obs_part - goal))
        else:
            raise Exception('If youre getting this exception, something is wrong with the code')


    def reset(self):
        if self.buffer is not None:

            if self.hindsight_strategy == 'final':
                self.do_final_strategy()
            elif self.hindsight_strategy == 'future':
                self.do_future_strategy(k=self.future_strategy_k)
            else:
                raise Exception('If youre getting this exception, something is wrong with the code')

            self.current_trajectory = []
        self.set_column_position(self.new_column_position())
        self.goal = self.new_goal()
        obs = self.get_observation()
        self.starting_position = np.copy(obs[:(self.num_factors if self.use_encoding else self.num_columns)])
        self.num_steps = 0
        return self.get_observation()

    def render(self):
        column_position = self.get_column_position()
        cv2.imshow('game', 255 * make_n_columns(column_position, spacing=2, size=128))
        cv2.waitKey(1)




# if __name__ == '__main__':
#     env = HighLevelColumnEnvironment(perfect_agents=True)
#     s = env.reset()
#     t = False
#     while True:
#         text_input = input('action:')
#         if text_input == "-1":
#             s = env.reset()
#             env.render()
#             continue
#         (action_num, goal) = re.match(r'^(\d+) ([\-\d\.]+)$', text_input).groups()
#         action_num = int(action_num)
#         goal = float(goal)
#         s, r, t, info = env.step((action_num, goal))
#         if t:
#             s = env.reset()
#         env.render()

if __name__ == '__main__':
    env = DummyHighLevelEnv(sparse_reward=True, num_columns=1)
    print(env.observation_space)
    s = env.reset()
    goal = s[1]
    sp, r, t, _ = env.step([0, goal], lambda x: x)
    print(sp, r, t)









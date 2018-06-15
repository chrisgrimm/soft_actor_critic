import re

import cv2
import numpy as np
from gym.spaces import Box

from goal_wrapper import GoalWrapper


def make_n_columns(height_percents, spacing=2, size=32):
    background = np.zeros((size, size, 3), dtype=np.float32)
    num_columns = len(height_percents)
    column_width = (size - (num_columns-1)*spacing) // num_columns
    assert column_width > 0
    start_pos = 0
    for i in range(num_columns):
        height = int(height_percents[i]*size)
        background[size-height:size, start_pos:start_pos+column_width, :] = 1
        start_pos += (column_width + spacing)
    return background

def get_batch_n_columns(batch_size, size=32, num_columns=3, spacing=2):
    images = []
    for i in range(batch_size):
        images.append(make_n_columns(np.random.uniform(0, 1, size=num_columns), spacing=spacing, size=size))
    return np.array(images)


class ColumnGame(object):

    def __init__(self, nn, num_columns=8, force_max=0.1, reward_per_goal=1.0, reward_no_goal=-0.01,
                 visual=True, indices=None):

        self.nn = nn
        # column image-generation settings
        self.image_size = 128
        self.resized_size = 32
        self.spacing = 2
        self.num_columns = num_columns
        self.goal_size = self.nn.hidden_size
        self.indices = indices

        self.force_max = force_max
        self.column_positions = np.array([0.5]*self.num_columns)
        self.visual = visual

        self.goal = self.generate_goal()
        self.goal_threshold = 0.1
        self.reward_per_goal = reward_per_goal
        self.reward_no_goal = reward_no_goal
        self.max_episode_steps = 100
        self.action_space = Box(low=-1, high=1, shape=[self.num_columns])
        if visual:
            self.observation_space = Box(low=0, high=1, shape=[self.resized_size, self.resized_size, 3+self.goal_size])
        else:
            self.observation_space = Box(low=-3, high=3, shape=[20+20])

        # if factor_number is -1, enforce all the goals. if factor number is 0-20

        self.episode_step = 0

    def generate_goal(self):
        image = make_n_columns(np.random.uniform(0, 1, size=self.num_columns), spacing=self.spacing, size=self.image_size)
        encoding = self.nn.encode_deterministic([image])[0]
        return encoding

    def get_observation(self):
        if self.visual:
            image = make_n_columns(self.column_positions, spacing=self.spacing, size=self.image_size)
            goal_image = np.tile(np.reshape(self.goal, [1, 1, self.goal_size]), [self.image_size, self.image_size, 1])
            combined_image = np.concatenate([image, goal_image], axis=2)
            return combined_image
        else:
            image = make_n_columns(self.column_positions, spacing=self.spacing, size=self.image_size)
            encoded = self.nn.encode_deterministic([image])[0]
            return np.concatenate([encoded, self.goal], axis=0)

    def resize_observation(self, obs):
        return cv2.resize(obs, (self.resized_size, self.resized_size), interpolation=cv2.INTER_NEAREST)

    def at_goal(self, vector, goal, indices=None):
        indices = range(self.goal_size) if indices is None else indices
        dist_to_goal = np.max(np.abs(vector[indices] - goal[indices]))
        return dist_to_goal < self.goal_threshold

    def compute_unnecessary_movement_penalty(self, vector, starting_vector, indices=None):
        # penalize changes to the features on the unselected indices.
        vector = np.copy(vector)
        vector[indices] = 0
        starting_vector = np.copy(starting_vector)
        starting_vector[indices] = 0
        return np.sum(np.abs(vector - starting_vector))

    # action is [-1, 1] x num_columns vector
    def step(self, action):
        #print('columns', self.column_positions)
        action = self.force_max * np.clip(action, -1, 1)
        self.episode_step += 1
        self.column_positions = np.clip(self.column_positions + action, 0, 1)
        #print('new_columns', self.column_positions)
        obs = self.get_observation()
        if self.visual:
            encoding = self.nn.encode_deterministic([obs[:, :, :3]])[0]
        else:
            encoding = obs[:20]
        at_goal = self.at_goal(encoding, self.goal, self.indices)
        reward = self.reward_per_goal if at_goal else self.reward_no_goal
        penalty = self.compute_unnecessary_movement_penalty(encoding, self.starting_vector, self.indices)
        reward = (reward - penalty)
        terminal = at_goal or (self.episode_step >= self.max_episode_steps)
        if self.visual:
            obs = self.resize_observation(obs)
        return obs, reward, terminal, {'vector': np.copy(self.column_positions)}

    def reset(self):
        self.column_positions = np.random.uniform(0, 1, size=self.num_columns)
        #self.column_positions = np.array([0.5]*self.num_columns)
        self.goal = self.generate_goal()
        self.episode_step = 0
        if self.visual:
            obs = self.get_observation()
            self.starting_vector = self.nn.encode_deterministic(obs[:, :, :3])
            obs = self.resize_observation(obs)
        else:
            obs = self.get_observation()
            self.starting_vector = obs[:20]
        return obs

    def render(self):
        if self.visual:
            cv2.imshow('game', 255*self.resize_observation(self.get_observation())[:, :, :3])
            cv2.waitKey(1)
        else:
            cv2.imshow('game', 255*make_n_columns(self.column_positions, spacing=self.spacing, size=self.image_size))
            cv2.waitKey(1)

class ColumnGameGoalWrapper(GoalWrapper):

    def __init__(self, env, buffer, reward_scaling, indices=None):
        self.env = env
        self.indices = range(self.env.goal_size) if indices is None else indices
        super(ColumnGameGoalWrapper, self).__init__(env, buffer, reward_scaling)

    def get_state_vector(self, obs_part):
        num_columns = self.env.num_columns
        size = self.env.image_size
        spacing = self.env.spacing
        column_width = (size - (num_columns - 1) * spacing) // num_columns

        height_percents = np.zeros(shape=[num_columns], dtype=np.float32)
        start_pos = 0
        for i in range(num_columns):
            indices = obs_part[:, start_pos, 0] == 1
            active_indices = np.linspace(0, 1, num=size)[indices]
            if len(active_indices) == 0:
                height_percents[i] = 0.0
            else:
                height_percents[i] = 1 - np.min(active_indices)
            start_pos += (column_width + spacing)
        return height_percents


    def obs_part_to_goal(self, obs_part):
        goal = np.zeros(shape=[self.env.goal_size])
        encoded = self.env.nn.encode_deterministic([obs_part[:, :, :3]])[0]
        goal[self.indices] = encoded[self.indices]
        return goal

    def at_goal(self, obs_part, goal):
        vector = np.zeros(shape=[self.env.goal_size])
        encoded = self.env.nn.encode_deterministic([obs_part[:, :, :3]])[0]
        vector[self.indices] = encoded[self.indices]
        dist = np.max(np.abs(vector - goal))
        return dist < self.env.goal_threshold

    def reward(self, obs_part, goal):
        return 1.0 if self.at_goal(obs_part, goal) else 0.0


    def terminal(self, obs_part, goal):
        return self.at_goal(obs_part, goal)

    def get_obs_part(self, obs):
        return obs[: ,:, :3+self.env.goal_size]

    def get_goal_part(self, obs):
        return np.reshape(obs[0, 0, 3+self.env.goal_size:], [self.env.goal_size])

    def obs_from_obs_part_and_goal(self, obs_part, goal):
        size = self.env.image_size
        goal_image = np.tile(np.reshape(goal, [1, 1, self.env.goal_size]), [size, size, 1])
        return np.concatenate([obs_part, goal_image], axis=2)

    def final_goal(self):
        goal = np.zeros(shape=[self.env.self.env.goal_size])
        goal[self.indices] = self.env.goal[self.indices]
        return np.array(goal)


def test_obs_to_vector(nn):
    env = ColumnGame(nn)
    env.reset()
    wrapped = ColumnGameGoalWrapper(env, None, 1.0)
    for i in range(1000):
        image = env.reset()
        ground_truth = env.column_positions
        recon_truth = wrapped.get_state_vector(image)
        dist = np.mean(np.abs(ground_truth - recon_truth))
        assert dist < 0.02

def run_game(nn):
    vae_index = 8
    action_index = 3
    env = ColumnGame(nn, indices=[vae_index], visual=False)
    s = env.reset()
    while True:
        command = input('command')
        print(command)
        match = re.match(r'^(\w) (\d+)$', command)
        if match:
            (command_type, number) = match.groups()
            number = int(number)
            if command_type == 'v':
                vae_index = number
                env.indices = [vae_index]
            elif command_type =='a':
                action_index = number
            else:
                print(f'Command Type {command_type} unrecognized')
            continue
        else:
            if command not in ['w', 's']:
                print(f'Unrecognized Action: {command}')
                continue
            action = np.zeros(shape=[env.num_columns])
            action[action_index] = 0.3 if command == 'w' else -0.3

        s, r, t, _ = env.step(action)
        print(s)
        print(r, 'terminal' if t else '')
        env.render()
        if t:
            s = env.reset()



if __name__ == '__main__':
    from indep_control2.vae_network import VAE_Network
    nn = VAE_Network(20, 128, 'image')
    nn.restore('./indep_control2/vae_network.ckpt')
    #test_obs_to_vector(nn)
    run_game(nn)


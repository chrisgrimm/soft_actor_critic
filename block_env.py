from gym import Env
from gym.spaces import Box, Discrete
from goal_wrapper import GoalWrapper

import numpy as np
import cv2
import os
PATH = os.path.split(os.path.realpath(__file__))[0]

def create_translated_uncolored_box(box_size=12, xy=None):
    out = np.zeros((28, 28, 3))
    #color = [1,0,0] if np.random.uniform(0, 1) < 0.5 else [0,1,0]
    color = [1,1,1]
    box = np.reshape(color, [1,1,3])
    if xy is None:
        x, y = np.random.randint(0, 28-box_size), np.random.randint(0, 28-box_size)
    else:
        x, y = xy[0], xy[1]
    out[y:y+box_size, x:x+box_size] = box
    return out



class BlockEnv(Env):

    def __init__(self):
        self.observation_space = Box(0, 1, [28, 28, 3])
        self.action_space = Discrete(4)
        self.num_steps = 1
        self.box_position = np.array([0, 0])
        self.action_dictionary = {0: np.array([1,0]),
                                  1: np.array([-1,0]),
                                  2: np.array([0, 1]),
                                  3: np.array([0, -1])}
        self.max_steps = 100
        self.box_size = 12
        self.step_number = 0


    def reset(self):
        self.box_position = np.random.randint(0, 28-self.box_size, size=2)
        self.step_number = 0
        image = create_translated_uncolored_box(self.box_size, xy=self.box_position.astype(np.int32))
        return image


    def render(self, *args, **kwargs):
        cv2.imshow('game', cv2.resize(255*create_translated_uncolored_box(self.box_size, xy=self.box_position.astype(np.int32)), (400, 400)))
        cv2.waitKey(1)

    def step(self, action):
        self.box_position = self.box_position + self.num_steps * self.action_dictionary[action]
        self.box_position = np.clip(self.box_position, 0, 28-self.box_size-1)
        image = create_translated_uncolored_box(box_size=self.box_size, xy=self.box_position.astype(np.int32))
        reward = 0
        self.step_number += 1
        terminal = self.step_number >= self.max_steps
        return image, reward, terminal, {}

from indep_control.networks import GANNetwork

class BlockGoalWrapper(GoalWrapper):

    def __init__(self, env, buffer, reward_scaling, factor, num_factors, factor_size):
        self.nn = GANNetwork()
        self.nn.restore(os.path.join(PATH, 'indep_control/weights.ckpt'))
        self.factor = factor
        self.num_factors = num_factors
        self.factor_size = factor_size
        self.reset_called()
        self.closeness_cutoff = 0.1
        super(BlockGoalWrapper, self).__init__(env, buffer, reward_scaling)

    def get_factor(self, obs_part):
        return self.nn.encode_single([obs_part])[0][self.factor * self.factor_size : (self.factor + 1) * self.factor_size]

    def encode_factor_as_image_channel(self, factor):
        tile_num = int(((28 * 28) / 10) + 1)
        return np.tile(np.reshape(factor, [1, 1, self.factor_size]), [28, 28, 1])
        #return np.reshape(np.tile(factor, tile_num)[:28*28], [28, 28, 1])

    def decode_factor_from_image_channel(self, image_channel):
        return np.reshape(image_channel[0, 0, :], [self.factor_size])

    def obs_part_to_goal(self, obs_part):
        goal = self.get_factor(obs_part)
        return goal

    def reset_called(self):
        print('Reset called!')
        xy = np.random.randint(0, 28-12, size=2)
        self.final_goal_image = create_translated_uncolored_box(box_size=12, xy=xy)
        self.final_goal_factor = self.get_factor(self.final_goal_image)


    def at_goal(self, obs_part, goal):
        obs_factor = self.get_factor(obs_part)
        dist = np.sqrt(np.sum(np.square(obs_factor - goal)))
        return dist < self.closeness_cutoff

    def reward(self, obs_part, goal):
        return 100 if self.at_goal(obs_part, goal) else -1

    def terminal(self, obs_part, goal):
        return self.at_goal(obs_part, goal)

    def get_obs_part(self, obs):
        image = obs[:, :, :3]
        return image

    def get_goal_part(self, obs):
        goal_channel_encoding = obs[:, :, 3:]
        goal = self.decode_factor_from_image_channel(goal_channel_encoding)
        return goal

    def obs_from_obs_part_and_goal(self, obs_part, goal):
        goal_channel_encoding = self.encode_factor_as_image_channel(goal)
        return np.concatenate([obs_part, goal_channel_encoding], axis=2)

    def final_goal(self):
        return self.final_goal_factor



if __name__ == '__main__':
    env = BlockEnv()
    s = env.reset()
    while True:
        a = np.random.randint(0, 4)
        s, r, t, info = env.step(a)
        env.render()
        print('boop!')
        if t:
            s = env.reset()





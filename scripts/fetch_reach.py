import argparse

import click
import gym
import numpy as np
import tensorflow as tf
from gym.envs.robotics import FetchReachEnv
from gym.envs.robotics.fetch_env import goal_distance

from environments.hindsight_wrapper import HindsightWrapper
from sac.train import HindsightTrainer
from scripts.gym_env import str_to_activation, cast_to_int

ACHIEVED_GOAL = 'achieved_goal'


class FetchReachHindsightWrapper(HindsightWrapper):
    def __init__(self, env):
        assert isinstance(env.unwrapped, FetchReachEnv)
        super().__init__(env)

    def achieved_goal(self, obs):
        return obs[ACHIEVED_GOAL]

    def reward(self, obs, goal):
        return self.env.compute_reward(obs[ACHIEVED_GOAL], goal, {})

    def at_goal(self, obs, goal):
        return goal_distance(obs[ACHIEVED_GOAL],
                             goal) < self.env.unwrapped.distance_threshold

    def desired_goal(self):
        return self.env.unwrapped.goal.copy()

    @staticmethod
    def vectorize_state(state):
        return np.concatenate([
            state.obs['achieved_goal'], state.obs['desired_goal'],
            state.obs['observation']
        ])


@click.option('--seed', default=0, type=int)
@click.option('--activation', default='relu', callback=str_to_activation)
@click.option('--n-layers', default=3, type=int)
@click.option('--layer-size', default=256, type=int)
@click.option('--learning-rate', default=3e-4, type=float)
@click.option('--buffer-size', default=1e7, callback=cast_to_int)
@click.option('--num-train-steps', default=4, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--reward-scale', default=9e3, type=float)
@click.option('--reward-prop', action='store_true')
@click.option('--logdir', default=None, type=str)
@click.option('--render', action='store_true')
def cli(seed, buffer_size, activation, n_layers, layer_size, learning_rate,
        reward_scale, batch_size, num_train_steps, logdir, render):
    # if args.mimic_file is not None:
    #     inject_mimic_experiences(args.mimic_file, buffer, N=10)

    HindsightTrainer(env=FetchReachHindsightWrapper(gym.make('FetchReach-v0')),
                     seed=seed,
                     buffer_size=buffer_size,
                     activation=activation,
                     n_layers=n_layers,
                     layer_size=layer_size,
                     learning_rate=learning_rate,
                     reward_scale=reward_scale,
                     batch_size=batch_size,
                     num_train_steps=num_train_steps,
                     logdir=logdir,
                     render=render)

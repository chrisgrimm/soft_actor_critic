import argparse

import gym
import numpy as np
import tensorflow as tf
from gym.envs.robotics import FetchReachEnv
from gym.envs.robotics.fetch_env import goal_distance

from environments.hindsight_wrapper import HindsightWrapper
from sac.train import HindsightTrainer, activation_type

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument(
        '--activation', default=tf.nn.relu, type=activation_type)
    parser.add_argument('--n-layers', default=3, type=int)
    parser.add_argument('--layer-size', default=256, type=int)
    parser.add_argument('--learning-rate', default=3e-4, type=float)
    parser.add_argument('--buffer-size', default=int(10**7), type=int)
    parser.add_argument('--num-train-steps', default=4, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--reward-scale', default=9e3, type=float)
    parser.add_argument('--mimic-file', default=None, type=str)
    parser.add_argument('--reward-prop', action='store_true')
    parser.add_argument('--logdir', default=None, type=str)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    # if args.mimic_file is not None:
    #     inject_mimic_experiences(args.mimic_file, buffer, N=10)

    HindsightTrainer(
        env=FetchReachHindsightWrapper(gym.make('FetchReach-v0')),
        seed=args.seed,
        buffer_size=args.buffer_size,
        activation=args.activation,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        learning_rate=args.learning_rate,
        reward_scale=args.reward_scale,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        logdir=args.logdir,
        render=args.render)

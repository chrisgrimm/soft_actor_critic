import argparse

import gym

from environment.goal_wrapper import MountaincarGoalWrapper
from sac.main import HindsightTrainer
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='HalfCheetah-v2')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--logdir', default=None, type=str)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    HindsightTrainer(
        env=MountaincarGoalWrapper(gym.make('MountainCarContinuous-v0')),
        seed=args.seed,
        buffer_size=10 ** 7,
        activation=tf.nn.relu,
        n_layers=3,
        layer_size=256,
        learning_rate=3e-4,
        reward_scale=10,
        batch_size=32,
        num_train_steps=1,
        logdir=args.logdir,
        render=args.render)

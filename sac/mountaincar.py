import argparse

import gym
import tensorflow as tf

from environment.hindsight_wrapper import MountaincarHindsightWrapper
from sac.train import HindsightPropagationTrainer, HindsightTrainer, activation_type

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='HalfCheetah-v2')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--reward-scale', default=1000, type=int)
    parser.add_argument('--activation', default=tf.nn.relu, type=activation_type)
    parser.add_argument('--logdir', default=None, type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save-path', default=None, type=str)
    parser.add_argument('--load-path', default=None, type=str)
    args = parser.parse_args()

    HindsightTrainer(
        env=MountaincarHindsightWrapper(gym.make('MountainCarContinuous-v0')),
        max_steps=999,
        seed=args.seed,
        buffer_size=10**7,
        activation=args.activation,
        n_layers=3,
        layer_size=256,
        learning_rate=3e-4,
        reward_scale=args.reward_scale,
        batch_size=32,
        num_train_steps=1,
        logdir=args.logdir,
        save_path=args.save_path,
        load_path=args.load_path,
        render=args.render)

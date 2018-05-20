import argparse

import gym
import tensorflow as tf

from environment.hindsight_wrapper import MountaincarHindsightWrapper
from sac.train import HindsightPropagationTrainer, HindsightTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='HalfCheetah-v2')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--default-reward', default=0, type=float)
    parser.add_argument('--logdir', default=None, type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save-path', default=None, type=str)
    parser.add_argument('--load-path', default=None, type=str)
    args = parser.parse_args()

    HindsightTrainer(
        env=MountaincarHindsightWrapper(gym.make('MountainCarContinuous-v0'),
                                        default_reward=args.default_reward),
        seed=args.seed,
        buffer_size=10**7,
        activation=tf.nn.relu,
        n_layers=3,
        layer_size=256,
        learning_rate=3e-4,
        grad_clip=None,
        reward_scale=1e3,
        batch_size=32,
        num_train_steps=1,
        logdir=args.logdir,
        save_path=args.save_path,
        load_path=args.load_path,
        render=args.render)

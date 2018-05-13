import argparse

import tensorflow as tf
from gym.wrappers import TimeLimit

from environment.hindsight_wrapper import PickAndPlaceHindsightWrapper
from environment.pick_and_place import PickAndPlaceEnv
from sac.train import (HindsightPropagationTrainer, HindsightTrainer,
                       activation_type)
import gym

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
    parser.add_argument('--max-steps', default=500, type=int)
    parser.add_argument('--geofence', default=.4, type=float)
    parser.add_argument('--min-lift-height', default=.02, type=float)
    parser.add_argument('--mimic-file', default=None, type=str)
    parser.add_argument('--random-block', action='store_true')
    parser.add_argument('--reward-prop', action='store_true')
    parser.add_argument('--logdir', default=None, type=str)
    parser.add_argument('--save-path', default=None, type=str)
    parser.add_argument('--load-path', default=None, type=str)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    # if args.mimic_file is not None:
    #     inject_mimic_experiences(args.mimic_file, buffer, N=10)
    trainer = HindsightPropagationTrainer if args.reward_prop else HindsightTrainer

    trainer(
        env=PickAndPlaceHindsightWrapper(
            TimeLimit(
                max_episode_steps=args.max_steps,
                env=PickAndPlaceEnv(
                    random_block=args.random_block,
                    min_lift_height=args.min_lift_height,
                    geofence=args.geofence))),
        seed=args.seed,
        buffer_size=int(args.buffer_size),
        activation=args.activation,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        learning_rate=args.learning_rate,
        reward_scale=args.reward_scale,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        logdir=args.logdir,
        save_path=args.save_path,
        load_path=args.load_path,
        render=args.render)

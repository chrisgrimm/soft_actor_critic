import click
import tensorflow as tf
from gym.wrappers import TimeLimit

from environments.hindsight_wrapper import PickAndPlaceHindsightWrapper
from environments.pick_and_place import PickAndPlaceEnv
from sac.train import HindsightPropagationTrainer, HindsightTrainer
from scripts.gym_env import cast_to_int, str_to_activation, check_probability


@click.command()
@click.option('--seed', default=0, type=int)
@click.option('--activation', default='relu', callback=str_to_activation)
@click.option('--n-layers', default=3, type=int)
@click.option('--layer-size', default=256, type=int)
@click.option('--learning-rate', default=2e-4, type=float)
@click.option('--buffer-size', default=1e7, callback=cast_to_int)
@click.option('--num-train-steps', default=4, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--reward-scale', default=9e3, type=float)
@click.option('--cheat-prob', default=0, type=float, callback=check_probability)
@click.option('--max-steps', default=500, type=int)
@click.option('--geofence', default=.4, type=float)
@click.option('--min-lift-height', default=.02, type=float)
@click.option('--default-reward', default=0, type=float)
@click.option('--grad-clip', default=1e6, type=float)
@click.option('--random-block', is_flag=True)
@click.option('--reward-prop', is_flag=True)
@click.option('--discrete', is_flag=True)
@click.option('--mimic-dir', default=None, type=str)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
@click.option('--render', is_flag=True)
def cli(reward_prop, default_reward, max_steps, discrete, random_block,
        min_lift_height, geofence, seed, buffer_size, activation, n_layers,
        layer_size, learning_rate, reward_scale, cheat_prob, grad_clip, batch_size,
        num_train_steps, mimic_dir, logdir, save_path, load_path, render):

    trainer = HindsightPropagationTrainer if reward_prop else HindsightTrainer

    trainer(
        env=PickAndPlaceHindsightWrapper(
            default_reward=default_reward,
            env=TimeLimit(
                max_episode_steps=max_steps,
                env=PickAndPlaceEnv(
                    discrete=discrete,
                    cheat_prob=cheat_prob,
                    random_block=random_block,
                    min_lift_height=min_lift_height,
                    geofence=geofence))),
        seed=seed,
        buffer_size=int(buffer_size),
        activation=activation,
        n_layers=n_layers,
        layer_size=layer_size,
        learning_rate=learning_rate,
        reward_scale=reward_scale,
        grad_clip=grad_clip,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        mimic_dir=mimic_dir,
        logdir=logdir,
        save_path=save_path,
        load_path=load_path,
        render=render)

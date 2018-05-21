import click
import gym
import tensorflow as tf

from sac.train import PropagationTrainer, Trainer


def cast_to_int(ctx, param, value):
    try:
        return int(value)
    except ValueError:
        raise click.BadParameter("Cannot cast param {} to int".format(value))


def check_probability(ctx, param, value):
    if not (0 <= value <= 1):
        raise click.BadParameter("Param {} should be between 0 and 1".format(value))
    return value


def str_to_activation(ctx, param, value):
    activations = dict(
        relu=tf.nn.relu,
        crelu=tf.nn.crelu,
        selu=tf.nn.selu,
        elu=tf.nn.elu,
        leaky=tf.nn.leaky_relu,
        leaky_relu=tf.nn.leaky_relu,
        tanh=tf.nn.tanh,
    )
    try:
        return activations[value]
    except KeyError:
        raise click.BadParameter(
            "Activation name must be one of the following:", '\n'.join(
                activations.keys()))


@click.command()
@click.option('--env', default='HalfCheetah-v2')
@click.option('--seed', default=0, type=int)
@click.option('--activation', default='relu', callback=str_to_activation)
@click.option('--n-layers', default=3, type=int)
@click.option('--layer-size', default=256, type=int)
@click.option('--learning-rate', default=3e-4, type=float)
@click.option('--buffer-size', default=1e7, callback=cast_to_int)
@click.option('--num-train-steps', default=1, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--reward-scale', default=1., type=float)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
@click.option('--render', is_flag=True)
@click.option('--reward-prop', is_flag=True)
def cli(reward_prop, env, seed, buffer_size, activation, n_layers, layer_size,
        learning_rate, reward_scale, batch_size, num_train_steps, logdir,
        save_path, load_path, render):
    # if args.mimic_file is not None:
    #     inject_mimic_experiences(args.mimic_file, buffer, N=10)

    trainer = PropagationTrainer if reward_prop else Trainer
    trainer(
        env=gym.make(env),
        seed=seed,
        buffer_size=buffer_size,
        activation=activation,
        n_layers=n_layers,
        layer_size=layer_size,
        learning_rate=learning_rate,
        reward_scale=reward_scale,
        batch_size=batch_size,
        grad_clip=None,
        num_train_steps=num_train_steps,
        logdir=logdir,
        save_path=save_path,
        load_path=load_path,
        render=render)

import argparse
import itertools
import pickle
import time
from collections import Counter

import gym
import numpy as np
import tensorflow as tf
from gym import spaces

from environment.goal_wrapper import (GoalWrapper, MountaincarGoalWrapper,
                                      PickAndPlaceGoalWrapper)
from environment.pick_and_place import PickAndPlaceEnv
from sac.agent import AbstractSoftActorCritic
from sac.chaser import ChaserEnv
from sac.policies import CategoricalPolicy, GaussianPolicy
from sac.replay_buffer.replay_buffer import ReplayBuffer2


def build_agent(env, activation, n_layers, layer_size, learning_rate):
    state_shape = env.observation_space.shape
    if isinstance(env.action_space, spaces.Discrete):
        action_shape = [env.action_space.n]
        PolicyType = CategoricalPolicy
    else:
        action_shape = env.action_space.shape
        PolicyType = GaussianPolicy

    class Agent(PolicyType, AbstractSoftActorCritic):
        def __init__(self, s_shape, a_shape):
            super(Agent, self).__init__(
                s_shape=s_shape,
                a_shape=a_shape,
                activation=activation,
                n_layers=n_layers,
                layer_size=layer_size,
                learning_rate=learning_rate)

    return Agent(state_shape, action_shape)


def inject_mimic_experiences(mimic_file, buffer, N=1):
    with open(mimic_file, 'rb') as f:
        mimic_trajectories = [pickle.load(f)]
    for trajectory in mimic_trajectories:
        for (s1, a, r, s2, t) in trajectory:
            for _ in range(N):
                buffer.append(s1=s1, a=a, r=r, s2=s2, t=t)


class Trainer:
    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def action_converter(self, action):
        """ Preprocess action before feeding to env """
        if type(self.env.action_space) is spaces.Discrete:
            return np.argmax(action)
        else:
            action = np.tanh(action)
            hi, lo = self.env.action_space.high, self.env.action_space.low
            return ((action + 1) / 2) * (hi - lo) + lo

    def state_converter(self, state):
        """ Preprocess state before feeding to network """
        return state

    def __init__(self, env, seed, buffer_size, activation, n_layers,
                 layer_size, learning_rate, reward_scale, batch_size,
                 num_train_steps, logdir, render):

        if seed is not None:
            np.random.seed(seed)
            tf.set_random_seed(seed)
            env.seed(seed)

        self.env = env
        self.buffer = ReplayBuffer2(buffer_size)
        self.reward_scale = reward_scale

        s1 = self.reset()

        agent = build_agent(
            env=env,
            activation=activation,
            n_layers=n_layers,
            layer_size=layer_size,
            learning_rate=learning_rate)

        tb_writer = None
        saver = tf.train.Saver()
        if logdir:
            tb_writer = tf.summary.FileWriter(
                logdir=logdir, graph=agent.sess.graph)

        count = Counter(reward=0, episode=0)
        episode_count = Counter()
        evaluation_period = 10

        for time_steps in itertools.count():
            is_eval_period = count['episode'] % evaluation_period == 0
            a = agent.get_actions(
                [self.state_converter(s1)], sample=(not is_eval_period))
            if render:
                env.render()
            s2, r, t, info = self.step(self.action_converter(a))
            if t:
                print('reward:', r)

            tick = time.time()

            episode_count += Counter(reward=r, timesteps=1)
            if is_eval_period:
                save_path = saver.save(agent.sess, "/tmp/model.ckpt")
                print("Model saved in path: {}".format(save_path))
            else:
                self.buffer.append(s1=s1, a=a, r=r * reward_scale, s2=s2, t=t)
                if len(self.buffer) >= batch_size:
                    for i in range(num_train_steps):
                        s1_sample, a_sample, r_sample, s2_sample, t_sample = self.buffer.sample(
                            batch_size)
                        s1_sample = list(map(self.state_converter, s1_sample))
                        s2_sample = list(map(self.state_converter, s2_sample))
                        [v_loss, q_loss, pi_loss] = agent.train_step(
                            s1_sample, a_sample, r_sample, s2_sample, t_sample)
                        episode_count += Counter({
                            'V loss': v_loss,
                            'Q loss': q_loss,
                            'pi loss': pi_loss
                        })
            s1 = s2
            if t:
                s1 = self.reset()
                print('({}) Episode {}\t Time Steps: {}\t Reward: {}'.format(
                    'EVAL' if is_eval_period else 'TRAIN', (count['episode']),
                    time_steps, episode_count['reward']))
                count += Counter(reward=(episode_count['reward']), episode=1)
                fps = int(episode_count['timesteps'] / (time.time() - tick))
                if logdir:
                    summary = tf.Summary()
                    if is_eval_period:
                        summary.value.add(
                            tag='eval reward',
                            simple_value=(episode_count['reward']))
                    summary.value.add(
                        tag='average reward',
                        simple_value=(
                            count['reward'] / float(count['episode'])))
                    summary.value.add(tag='fps', simple_value=fps)
                    for k in ['V loss', 'Q loss', 'pi loss', 'reward']:
                        summary.value.add(tag=k, simple_value=episode_count[k])
                    tb_writer.add_summary(summary, count['episode'])
                    tb_writer.flush()

                for k in episode_count:
                    episode_count[k] = 0


class HindsightTrainer(Trainer):
    def __init__(self, env, seed, buffer_size, reward_scale, batch_size,
                 num_train_steps, logdir, render, activation, n_layers,
                 layer_size, learning_rate):
        assert isinstance(env, GoalWrapper)
        self.trajectory = []
        super().__init__(
            env=env,
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
        self.s1 = self.reset()

    def step(self, action):
        s2, r, t, i = super().step(action)
        self.trajectory.append((self.s1, action, r, s2, t))
        self.s1 = s2
        return s2, r, t, i

    def reset(self):
        for s1, a, r, s2, t in self.env.recompute_trajectory(self.trajectory):
            self.buffer.append(s1=s1, a=a, r=r * self.reward_scale, s2=s2, t=t)
        self.trajectory = []
        self.s1 = super().reset()
        return self.s1

    def state_converter(self, state):
        return self.env.obs_from_obs_part_and_goal(state)


def activation(name):
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
        return activations[name]
    except KeyError:
        raise argparse.ArgumentTypeError(
            "Activation name must be one of the following:", '\n'.join(
                activations.keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='HalfCheetah-v2')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--activation', default=tf.nn.relu, type=activation)
    parser.add_argument('--n-layers', default=3, type=int)
    parser.add_argument('--layer-size', default=256, type=int)
    parser.add_argument('--learning-rate', default=3e-4, type=float)
    parser.add_argument('--buffer-size', default=int(10**7), type=int)
    parser.add_argument('--num-train-steps', default=1, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--reward-scale', default=1., type=float)
    parser.add_argument('--mimic-file', default=None, type=str)
    parser.add_argument('--logdir', default=None, type=str)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    # if args.mimic_file is not None:
    #     inject_mimic_experiences(args.mimic_file, buffer, N=10)

    Trainer(
        env=gym.make(args.env),
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

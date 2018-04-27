import argparse

import itertools
from collections import Counter

import numpy as np
import gym
import time

from environment.pick_and_place import PickAndPlaceEnv
from gym import spaces
from goal_wrapper import MountaincarGoalWrapper, PickAndPlaceGoalWrapper, GoalWrapper
import tensorflow as tf

from sac.replay_buffer.replay_buffer import ReplayBuffer2
from sac.networks.policy_mixins import MLPPolicy, GaussianPolicy, CategoricalPolicy
from sac.networks.value_function_mixins import MLPValueFunc
from sac.networks.network_interface import AbstractSoftActorCritic
from sac.chaser import ChaserEnv
import pickle


def build_agent(env):
    state_shape = env.observation_space.shape
    if type(env.action_space) is spaces.Discrete:
        action_shape = [env.action_space.n]
        PolicyType = CategoricalPolicy
    else:
        action_shape = env.action_space.shape
        PolicyType = GaussianPolicy

    class Agent(PolicyType, MLPPolicy, MLPValueFunc, AbstractSoftActorCritic):
        def __init__(self, s_shape, a_shape):
            super(Agent, self).__init__(s_shape, a_shape)

    return Agent(state_shape, action_shape)


def inject_mimic_experiences(mimic_file, buffer, N=1):
    with open(mimic_file, 'rb') as f:
        mimic_trajectories = [pickle.load(f)]
    for trajectory in mimic_trajectories:
        for (s1, a, r, s2, t) in trajectory:
            for _ in range(N):
                buffer.append(s1, a, r, s2, t)


def build_action_converter(env):
    def converter(a):
        if type(env.action_space) is spaces.Discrete:
            return np.argmax(a)
        else:
            a = np.tanh(a)
            h, l = env.action_space.high, env.action_space.low
            return ((a + 1) / 2) * (h - l) + l

    return converter


def string_to_env(env_name):
    using_hindsight = False
    if env_name == 'chaser':
        env = ChaserEnv()
    elif env_name == 'mountaincar-continuous-hindsight':
        env = MountaincarGoalWrapper(gym.make('MountainCarContinuous-v0'))
        using_hindsight = True
    elif env_name == 'pick-and-place':
        env = PickAndPlaceGoalWrapper(PickAndPlaceEnv(max_steps=500, neg_reward=False))
        using_hindsight = True
    else:
        env = gym.make(env_name)
    return env, using_hindsight


def run_training(env, buffer, reward_scale, batch_size, num_train_steps, logdir=None):
    V_LOSS = 'V loss'
    Q_LOSS = 'Q loss'
    PI_LOSS = 'pi loss'
    EPISODE = 'episode'
    REWARD = 'reward'

    tb_writer = tf.summary.FileWriter(logdir) if logdir else None

    s1 = env.reset()

    agent = build_agent(env)
    action_converter = build_action_converter(env)

    count = Counter(reward=0, episode=0)
    episode_count = Counter()
    evaluation_period = 10
    is_eval_period = lambda episode_number: episode_number % evaluation_period == 0

    for time_steps in itertools.count():
        a = agent.get_actions([s1], sample=(not is_eval_period(count[EPISODE])))

        a = a[0]
        s2, r, t, info = env.step(action_converter(a))
        if t:
            print('reward:', r)

        tick = time.time()

        episode_count += Counter(reward=r, timesteps=1)
        r *= reward_scale
        if not is_eval_period(count[EPISODE]):
            buffer.append(s1, action_converter(a), r, s2, t)
            if len(buffer) >= batch_size:
                for i in range(num_train_steps):
                    s1_sample, a_sample, r_sample, s2_sample, t_sample = buffer.sample(batch_size)
                    [v_loss, q_loss, pi_loss] = agent.train_step(s1_sample, a_sample, r_sample, s2_sample, t_sample)
                    episode_count += Counter({V_LOSS: v_loss, Q_LOSS: q_loss, PI_LOSS: pi_loss})
        s1 = s2
        if t:
            if isinstance(env, GoalWrapper):
                for s1, a, r, s2, t in env.recompute_trajectory():
                    buffer.append(s1, a, r * reward_scale, s2, t)
            s1 = env.reset()
            episode_reward = episode_count[REWARD]
            print('(%s) Episode %s\t Time Steps: %s\t Reward: %s' % ('EVAL' if is_eval_period(
                count[EPISODE]) else 'TRAIN',
                                                                     (count[EPISODE]), time_steps, episode_reward))
            count += Counter(reward=episode_reward, episode=1)
            fps = int(episode_count['timesteps'] / (time.time() - tick))
            if logdir:
                summary = tf.Summary()
                summary.value.add(tag='average reward', simple_value=count[REWARD] / float(count[EPISODE]))
                summary.value.add(tag='fps', simple_value=fps)
                for k in [V_LOSS, Q_LOSS, PI_LOSS, REWARD]:
                    summary.value.add(tag=k, simple_value=episode_count[k])
                tb_writer.add_summary(summary, count[EPISODE])
                tb_writer.flush()

            for k in episode_count:
                episode_count[k] = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='HalfCheetah-v2')
    parser.add_argument('--buffer-size', default=int(10 ** 7), type=int)
    parser.add_argument('--num-train-steps', default=1, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--reward-scale', default=10., type=float)
    parser.add_argument('--mimic-file', default=None, type=str)
    parser.add_argument('--logdir', default=None, type=str)
    args = parser.parse_args()

    buffer = ReplayBuffer2(args.buffer_size)
    env, using_hindsight = string_to_env(args.env)

    if args.mimic_file is not None:
        inject_mimic_experiences(args.mimic_file, buffer, N=10)
    run_training(env=env,
                 buffer=buffer,
                 reward_scale=args.reward_scale,
                 batch_size=args.batch_size,
                 num_train_steps=args.num_train_steps,
                 logdir=args.logdir)

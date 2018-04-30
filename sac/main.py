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
    if isinstance(env.action_space, spaces.Discrete):
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
                buffer.append(s1=s1, a=a, r=r, s2=s2, t=t)


def string_to_env(env_name):
    if env_name == 'chaser':
        return ChaserEnv()
    elif env_name == 'mountaincar':
        return MountaincarGoalWrapper(gym.make('MountainCarContinuous-v0'))
    elif env_name == 'pick-and-place':
        return PickAndPlaceGoalWrapper(
            PickAndPlaceEnv(max_steps=500, neg_reward=False))
    return gym.make(env_name)


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

    def __init__(self, env, buffer, reward_scale, batch_size, num_train_steps,
                 logdir, render):
        tb_writer = tf.summary.FileWriter(logdir) if logdir else None

        self.env = env
        self.buffer = buffer
        self.reward_scale = reward_scale

        s1 = self.reset()

        agent = build_agent(env)

        count = Counter(reward=0, episode=0)
        episode_count = Counter()
        evaluation_period = 10

        for time_steps in itertools.count():
            is_eval_period = count['episode'] % evaluation_period == 0
            a = agent.get_actions(
                [self.state_converter(s1)],
                sample=(not is_eval_period))
            if render:
                env.render()
            s2, r, t, info = self.step(self.action_converter(a))
            if t:
                print('reward:', r)

            tick = time.time()

            episode_count += Counter(reward=r, timesteps=1)
            if not is_eval_period:
                buffer.append(s1=s1, a=a, r=r * reward_scale, s2=s2, t=t)
                if len(buffer) >= batch_size:
                    for i in range(num_train_steps):
                        s1_sample, a_sample, r_sample, s2_sample, t_sample = buffer.sample(
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
                episode_reward = episode_count['reward']
                print('(%s) Episode %s\t Time Steps: %s\t Reward: %s' %
                      ('EVAL' if is_eval_period else 'TRAIN',
                       (count['episode']), time_steps, episode_reward))
                count += Counter(reward=episode_reward, episode=1)
                fps = int(episode_count['timesteps'] / (time.time() - tick))
                if logdir and is_eval_period:
                    summary = tf.Summary()
                    summary.value.add(
                        tag='average reward',
                        simple_value=count['reward'] / float(count['episode']))
                    summary.value.add(tag='fps', simple_value=fps)
                    for k in ['V loss', 'Q loss', 'pi loss', 'reward']:
                        summary.value.add(tag=k, simple_value=episode_count[k])
                    tb_writer.add_summary(summary, count['episode'])
                    tb_writer.flush()

                for k in episode_count:
                    episode_count[k] = 0


class HindsightTrainer(Trainer):
    def __init__(self, env, buffer, reward_scale, batch_size, num_train_steps,
                 logdir, render):
        assert isinstance(env, GoalWrapper)
        self.trajectory = []
        super().__init__(env, buffer, reward_scale, batch_size,
                         num_train_steps, logdir, render)
        self.s1 = self.reset()

    def step(self, action):
        s2, r, t, i = super().step(action)
        self.trajectory.append((self.s1, action, r, s2, t))
        self.s1 = s2
        return s2, r, t, i

    def reset(self):
        for s1, a, r, s2, t in env.recompute_trajectory(self.trajectory):
            self.buffer.append(s1=s1, a=a, r=r * self.reward_scale, s2=s2, t=t)
        self.trajectory = []
        self.s1 = super().reset()
        return self.s1

    def state_converter(self, state):
        return self.env.obs_from_obs_part_and_goal(state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='HalfCheetah-v2')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--buffer-size', default=int(10**7), type=int)
    parser.add_argument('--num-train-steps', default=1, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--reward-scale', default=1., type=float)
    parser.add_argument('--mimic-file', default=None, type=str)
    parser.add_argument('--logdir', default=None, type=str)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    buffer = ReplayBuffer2(args.buffer_size)
    env = string_to_env(args.env)

    if args.mimic_file is not None:
        inject_mimic_experiences(args.mimic_file, buffer, N=10)
    trainer = HindsightTrainer if isinstance(env, GoalWrapper) else Trainer
    trainer(
        env=env,
        buffer=buffer,
        reward_scale=args.reward_scale,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        logdir=args.logdir,
        render=args.render)

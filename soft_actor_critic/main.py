import argparse

import numpy as np
import gym
from gym import spaces

from replay_buffer.replay_buffer import ReplayBuffer2
from networks.policy_mixins import MLPPolicy, GaussianPolicy, CategoricalPolicy
from networks.value_function_mixins import MLPValueFunc
from networks.network_interface import AbstractSoftActorCritic


def build_agent(env):
    state_shape = env.observation_space.shape
    if type(env.action_space) is spaces.Discrete:
        print('is discrete')
        action_shape = [env.action_space.n]
        PolicyType = CategoricalPolicy
    else:
        print('is gaussian')
        action_shape = env.action_space.shape
        PolicyType = GaussianPolicy

    class Agent(PolicyType, MLPPolicy, MLPValueFunc, AbstractSoftActorCritic):
        def __init__(self, s_shape, a_shape):
            super(Agent, self).__init__(s_shape, a_shape)

    return Agent(state_shape, action_shape)


def build_action_converter(env):
    def converter(a):
        if type(env.action_space) is spaces.Discrete:
                return np.argmax(a)
        else:
            a = np.tanh(a)
            h, l = env.action_space.high, env.action_space.low
            return ((a + 1) / 2) * (h - l) + l
    return converter


def run_training(env, buffer_size, reward_scale, batch_size, num_train_steps):
    if env == 'chaser':
        env = ChaserEnv()
    else:
        env = gym.make(env)

    s1 = env.reset()

    agent = build_agent(env)
    action_converter = build_action_converter(env)

    buffer = ReplayBuffer2(buffer_size)
    episode_reward = 0
    episodes = 0
    time_steps = 0
    while True:
        a = agent.sample_actions([s1])
        a = a[0]
        # s2, r, t, info = env.step(2*a)
        # s2, r, t, info = env.step(np.argmax(a))
        s2, r, t, info = env.step(action_converter(a))
        time_steps += 1

        episode_reward += r
        # env.render()
        r /= reward_scale
        # print(s1)
        buffer.append(s1, a, r, s2, t)
        if len(buffer) >= batch_size:
            for i in range(num_train_steps):
                s1_sample, a_sample, r_sample, s2_sample, t_sample = buffer.sample(batch_size)
                [v_loss, q_loss, pi_loss] = agent.train_step(s1_sample, a_sample, r_sample, s2_sample, t_sample)
                # print('v_loss', v_loss, 'q_loss', q_loss, 'pi_loss', pi_loss)
        s1 = s2
        if t:
            s1 = env.reset()
            # summary = tf.Summary()
            # summary.value.add(tag='episode reward', simple_value=episode_reward)
            # agent.tb_writer.add_summary(summary, episodes)
            # agent.tb_writer.flush()
            print('Episode %s\t Time Steps: %s\t Reward: %s' % (episodes, time_steps, episode_reward))
            episode_reward = 0
            episodes += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='HalfCheetah-v2')
    parser.add_argument('--buffer-size', default=int(10 ** 7), type=int)
    parser.add_argument('--num-train-steps', default=1, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--reward-scale', default=1/10., type=float)
    args = parser.parse_args()
    run_training(env=args.env,
                 buffer_size=args.buffer_size,
                 reward_scale=args.reward_scale,
                 batch_size=args.batch_size,
                 num_train_steps=args.num_train_steps)

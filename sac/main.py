import argparse

import dill as pickle
import gym
import numpy as np
from gym import spaces
from indep_control2.utils import build_directory_structure, LOG
import os

from chaser import ChaserEnv
from column_game import ColumnGame
from indep_control2.vae_network import VAE_Network
from networks.network_interface import AbstractSoftActorCritic
from networks.policy_mixins import MLPPolicy, GaussianPolicy, CategoricalPolicy, CNN_Goal_Policy, CNN_Power2_Policy
from networks.value_function_mixins import MLPValueFunc, CNN_Goal_ValueFunc, CNN_Power2_ValueFunc
from replay_buffer.replay_buffer import ReplayBuffer2


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

def build_image_goal_agent(env):
    assert type(env.action_space) is spaces.Discrete
    action_shape = [env.action_space.n]
    PolicyType = CategoricalPolicy
    class Agent(PolicyType, CNN_Goal_Policy, CNN_Goal_ValueFunc, AbstractSoftActorCritic):
        def __init__(self, s_shape, a_shape):
            super(Agent, self).__init__(s_shape, a_shape)

    return Agent([28, 28, 3+10], action_shape)

def build_column_agent(env):
    class Agent(GaussianPolicy, CNN_Power2_Policy, CNN_Power2_ValueFunc, AbstractSoftActorCritic):
        def __init__(self, s_shape, a_shape):
            super(Agent, self).__init__(s_shape, a_shape)

    return Agent(env.observation_space.shape, env.action_space.shape)





def inject_mimic_experiences(mimic_file, buffer, N=1):
    with open(mimic_file, 'r') as f:
        mimic_trajectories = pickle.load(f)
    for trajectory in mimic_file:
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

def string_to_env(env_name, buffer, reward_scaling):
    if env_name == 'chaser':
        env = ChaserEnv()
    elif env_name == 'mountaincar_continuous_hindsight':
        env = MountaincarGoalWrapper(gym.make('MountainCarContinuous-v0'), buffer, reward_scaling=reward_scaling)
    else:
        env = gym.make(env_name)
    return env




def run_training(env, agent, buffer, reward_scale, batch_size, num_train_steps, hindsight_agent=False):
    s1 = env.reset()

    action_converter = build_action_converter(env)
    episode_reward = 0
    episodes = 0
    time_steps = 0
    episode_time_steps = 0
    evaluation_period = 10
    is_eval_period = lambda episode_number: episode_number % evaluation_period == 0
    while True:
        a = agent.get_actions([s1], sample=(not is_eval_period(episodes)))
        a = a[0]
        if hindsight_agent:
            s2, r, t, info = env.step(a, action_converter)
        else:
            s2, r, t, info = env.step(action_converter(a))

        time_steps += 1
        episode_time_steps += 1

        episode_reward += r
        env.render()
        r /= reward_scale
        if not is_eval_period(episodes):
            buffer.append(s1, a, r, s2, t)
            if len(buffer) >= batch_size:
                for i in range(num_train_steps):
                    s1_sample, a_sample, r_sample, s2_sample, t_sample = buffer.sample(batch_size)
                    [v_loss, q_loss, pi_loss] = agent.train_step(s1_sample, a_sample, r_sample, s2_sample, t_sample)
                    LOG.add_line('v_loss', v_loss)
                    LOG.add_line('q_loss', q_loss)
                    LOG.add_line('pi_loss', pi_loss)
        s1 = s2
        if t:
            s1 = env.reset()
            print('(%s) Episode %s\t Time Steps: %s\t Reward: %s' % ('EVAL' if is_eval_period(episodes) else 'TRAIN',
                                                                     episodes, time_steps, episode_reward))
            LOG.add_line('episode_reward', episode_reward)
            LOG.add_line('episode_length', episode_time_steps)
            episode_reward = 0
            episode_time_steps = 0
            episodes += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--env', default='HalfCheetah-v2')
    parser.add_argument('--buffer-size', default=int(10 ** 7), type=int)
    parser.add_argument('--num-train-steps', default=1, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--reward-scale', default=1., type=float)
    parser.add_argument('--run-name', type=str)
    parser.add_argument('--mimic-file', default=None, type=str)
    args = parser.parse_args()

    data_storage_dir = {args.run_name: {
        'data': {}
    }}
    build_directory_structure('.', data_storage_dir)
    base_path = os.path.join(args.run_name, 'data')
    LOG.setup({
        'episode_length': os.path.join(base_path, 'episode_length'),
        'episode_reward': os.path.join(base_path, 'episode_rewards'),
        'q_loss': os.path.join(base_path, 'q_loss'),
        'v_loss': os.path.join(base_path, 'v_loss'),
        'pi_loss': os.path.join(base_path, 'pi_loss')
    })



    buffer = ReplayBuffer2(args.buffer_size)


    #env = string_to_env(args.env, buffer, args.reward_scale)
    nn = VAE_Network(hidden_size=20, input_size=100, mode='image')
    nn.restore('./indep_control2/vae_network.ckpt')
    factor_num = 8
    env = ColumnGame(nn, indices=[factor_num])
    #env = BlockGoalWrapper(BlockEnv(), buffer, args.reward_scale, 0, 2, 10)
    agent = build_column_agent(env)

    #if args.mimic_file is not None:
    #    inject_mimic_experiences(args.mimic_file, buffer)
    run_training(env=env,
                 agent=agent,
                 buffer=buffer,
                 reward_scale=args.reward_scale,
                 batch_size=args.batch_size,
                 num_train_steps=args.num_train_steps,
                 hindsight_agent=False)

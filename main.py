import numpy as np
import gym
from chaser import ChaserEnv
from gym import spaces
from goal_wrapper import MountaincarGoalWrapper
import tensorflow as tf

from replay_buffer.replay_buffer import ReplayBuffer2

from networks.network_policy_mixins import GaussianPolicy, CategoricalPolicy
from networks.network_architecture_mixins import MLPPolicy, MLPPolicyGen, CNNPolicy
from networks.network_value_mixins import MLPValueFunc

from networks.network_interface import AbstractSoftActorCritic

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


def build_action_converter(env):
    if type(env.action_space) is spaces.Discrete:
        def converter(a):
            return np.argmax(a)
    else:
        def converter(a):
            a = np.tanh(a)
            h, l = env.action_space.high, env.action_space.low
            return ((a + 1) / 2) * (h - l) + l
    return converter



def run_training(env, buffer, reward_scale):
    num_train_steps = 4
    batch_size = 32
    s1 = env.reset()

    agent = build_agent(env)
    action_converter = build_action_converter(env)

    episode_reward = 0
    episodes = 0
    time_steps = 0
    evaluation_period = 10
    is_eval_period = lambda episode_number: episode_number % evaluation_period == 0
    while True:
        a = agent.get_actions([s1], sample=(not is_eval_period(episodes)))
        a = a[0]
        s2, r, t, info = env.step(action_converter(a))
        time_steps += 1

        episode_reward += r#info['base_reward']
        #env.render()
        r /= reward_scale
        if not is_eval_period(episodes):
            buffer.append(s1, a, r, s2, t)
            if len(buffer) >= batch_size:
                for i in range(num_train_steps):
                    s1_sample, a_sample, r_sample, s2_sample, t_sample = buffer.sample(batch_size)
                    [v_loss, q_loss, pi_loss] = agent.train_step(s1_sample, a_sample, r_sample, s2_sample, t_sample)
        s1 = s2
        if t:
            s1 = env.reset()
            print('(%s) Episode %s\t Time Steps: %s\t Reward: %s' % ('EVAL' if is_eval_period(episodes) else 'TRAIN',
                                                                     episodes, time_steps, episode_reward))
            episode_reward = 0
            episodes += 1


#env = gym.make('Ant-v2')
#env = gym.make('HalfCheetah-v2')
#env = gym.make('CartPole-v0')
#env = gym.make('Pendulum-v0')
#env = ChaserEnv()
buffer_size = 10**7
reward_scaling = 1./10
buffer = ReplayBuffer2(buffer_size)
env = MountaincarGoalWrapper(gym.make('MountainCarContinuous-v0'), buffer, reward_scaling=reward_scaling)
run_training(env, buffer, reward_scale=reward_scaling)


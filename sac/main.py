import argparse

import dill as pickle
import gym
import numpy as np
from gym import spaces
from indep_control2.utils import build_directory_structure, LOG
import os
import tensorflow as tf
from chaser import ChaserEnv
from column_game import ColumnGame
from indep_control2.vae_network import VAE_Network
from networks.network_interface import AbstractSoftActorCritic
from networks.policy_mixins import MLPPolicy, GaussianPolicy, CategoricalPolicy, CNN_Goal_Policy, CNN_Power2_Policy, \
    Categorical_X_GaussianPolicy
from networks.value_function_mixins import MLPValueFunc, CNN_Goal_ValueFunc, CNN_Power2_ValueFunc, \
    MLP_Categorical_X_Gaussian_ValueFunc
from replay_buffer.replay_buffer import ReplayBuffer2
from high_level_environment import HighLevelColumnEnvironment, DummyHighLevelEnv
from utils import get_best_gpu


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

def build_column_agent(env, name='SAC'):
    if env.visual:
        class Agent(GaussianPolicy, CNN_Power2_Policy, CNN_Power2_ValueFunc, AbstractSoftActorCritic):
            def __init__(self, s_shape, a_shape):
                super(Agent, self).__init__(s_shape, a_shape, global_name=name)
    else:
        class Agent(GaussianPolicy, MLPPolicy, MLPValueFunc, AbstractSoftActorCritic):
            def __init__(self, s_shape, a_shape):
                super(Agent, self).__init__(s_shape, a_shape, global_name=name)

    return Agent(env.observation_space.shape, env.action_space.shape)


# def build_high_level_agent(env, name='SAC_high_level'):
#     class Agent(Categorical_X_GaussianPolicy, MLPPolicy, MLP_Categorical_X_Gaussian_ValueFunc, AbstractSoftActorCritic):
#         def __init__(self, s_shape, a_shape):
#             super(Agent, self).__init__(s_shape, a_shape, global_name=name)
#     return Agent(env.env.observation_space.shape, [9])

def build_high_level_agent(env, name='SAC_high_level', learning_rate=1*10**-4, width=128, random_goal=False):
    class Agent(
        GaussianPolicy,
        MLPPolicy(width),
        MLPValueFunc(width),
        AbstractSoftActorCritic):
        def __init__(self, s_shape, a_shape):
            super(Agent, self).__init__(s_shape, a_shape, global_name=name, learning_rate=learning_rate, inject_goal_randomness=random_goal)
    return Agent(env.observation_space.shape, env.action_space.shape)

# def build_high_level_action_converter(env):
#     def converter(a):
#         a_cat, a_gauss = a[:8], a[8:]
#         a_cat = np.argmax(a_cat)
#         a_gauss = np.tanh(a_gauss)
#         #h, l = 2.5, -2.5
#         h, l = 1.0, 0.0
#         a_gauss = ((a_gauss + 1) / 2) * (h - l) + l
#         return (a_cat, a_gauss)
#     return converter

def build_high_level_action_converter(env):
    def converter(a):
        a_cat, a_gauss = a[0], a[1]
        a_cat = np.tanh(a_cat)
        a_cat = int((a_cat + 1) / 2.0 * 8)
        # handles stupid case when
        if a_cat == 8:
            a_cat = 7
        a_gauss = np.tanh(a_gauss)
        #h, l = 2.5, -2ƒ.5
        h, l = 1.0, 0.0
        a_gauss = ((a_gauss + 1) / 2) * (h - l) + l
        return (a_cat, a_gauss)
    return converter

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




def run_training(env, agent, buffer, reward_scale, batch_size, num_train_steps, hindsight_agent=False, run_name='', render=False):
    s1 = env.reset()

    #action_converter = build_action_converter(env)
    action_converter = build_high_level_action_converter(env)
    episode_reward = 0
    episodes = 0
    time_steps = 0
    episode_time_steps = 0
    evaluation_period = 100
    save_period = 1000
    is_eval_period = lambda episode_number: episode_number % evaluation_period == 0
    #is_eval_period = lambda episode_number: True
    while True:
        a = agent.get_actions([s1], sample=(not is_eval_period(episodes)))[0]

        if hindsight_agent:
            s2, r, t, info = env.step(a, action_converter)
        else:
            s2, r, t, info = env.step(a, action_converter)

        time_steps += 1
        episode_time_steps += 1

        episode_reward += r
        if render:
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
            print('TERMINAL')
            s1 = env.reset()
            #print('(%s) Episode %s\t Time Steps: %s\t Reward: %s' % ('EVAL' if is_eval_period(episodes) else 'TRAIN',
            #                                                         episodes, time_steps, episode_reward))
            print(f'{"EVAL" if is_eval_period(episodes) else "TRAIN"}\t Episode: {episodes}\t Time Steps: {episode_time_steps} Reward: {episode_reward}')
            LOG.add_line('episode_reward', episode_reward)
            LOG.add_line('episode_length', episode_time_steps)
            episode_reward = 0
            episode_time_steps = 0
            episodes += 1
            if episodes % save_period == 0:
                agent.save(os.path.join('.', 'runs', run_name, 'weights', 'sac.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--env', default='HalfCheetah-v2')
    parser.add_argument('--buffer-size', default=int(10 ** 6), type=int)
    parser.add_argument('--num-train-steps', default=4, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--reward-scale', default=0.01, type=float)
    parser.add_argument('--learning-rate', type=float, default=1*10**-4)
    parser.add_argument('--run-name', type=str)
    parser.add_argument('--mimic-file', default=None, type=str)
    parser.add_argument('--force-max', default=0.3, type=float)
    parser.add_argument('--reward-per-goal', default=10.0, type=float)
    parser.add_argument('--reward-no-goal', default=-0.01, type=float)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--factor-num', type=int)
    parser.add_argument('--use-encoding', action='store_true')
    parser.add_argument('--distance-mode', type=str)
    parser.add_argument('--gpu-num', type=int, default=-1)
    parser.add_argument('--network-width', type=int, default=128)
    parser.add_argument('--random-goal', action='store_true')
    parser.add_argument('--hindsight-strategy', type=str, default='final')
    parser.add_argument('--num-columns', type=int, default=8)

    args = parser.parse_args()

    data_storage_dir = {'runs': {
        args.run_name: {
            'data': {},
            'weights': {},
        },
    }}

    base_path = os.path.join('./runs', args.run_name, 'data')
    LOG.setup({
        'episode_length': os.path.join(base_path, 'episode_length'),
        'episode_reward': os.path.join(base_path, 'episode_rewards'),
        'q_loss': os.path.join(base_path, 'q_loss'),
        'v_loss': os.path.join(base_path, 'v_loss'),
        'pi_loss': os.path.join(base_path, 'pi_loss')
    })

    if os.path.isdir(os.path.join('./runs', args.run_name)):
        cmd = input(f'Run: {args.run_name} already exists. Purge? (Y/N)')
        if cmd in ['y', 'Y']:
            LOG.purge()
        else:
            raise Exception(f'Run: {args.run_name} already exists.')

    build_directory_structure('.', data_storage_dir)


    buffer = ReplayBuffer2(args.buffer_size)


    #env = string_to_env(args.env, buffer, args.reward_scale)
    #nn = VAE_Network(hidden_size=20, input_size=100, mode='image')
    #nn.restore('./indep_control2/vae_network.ckpt')
    #factor_num = args.factor_num
    #env = ColumnGame(nn, indices=[factor_num], force_max=args.force_max, reward_per_goal=args.reward_per_goal,
    #                 reward_no_goal=args.reward_no_goal, visual=False)
    #env = HighLevelColumnEnvironment(perfect_agents=True, buffer=buffer)
    env = DummyHighLevelEnv(sparse_reward=True, buffer=buffer, goal_reward=args.reward_per_goal,
                            use_encoding=args.use_encoding, distance_mode=args.distance_mode, hindsight_strategy=args.hindsight_strategy,
                            num_columns=args.num_columns)
    #env = gym.make('CartPole-v0')

    #env = BlockGoalWrapper(BlockEnv(), buffer, args.reward_scale, 0, 2, 10)
    #agent = build_column_agent(env)

    gpu_num = get_best_gpu() if args.gpu_num == -1 else args.gpu_num
    with tf.device(f'/gpu:{gpu_num}'):
        agent = build_high_level_agent(env, learning_rate=args.learning_rate, width=args.network_width, random_goal=args.random_goal)
    #agent = build_agent(env)
    if args.restore:
        restore_path = os.path.join('.', 'runs',  args.run_name, 'weights', 'sac.ckpt')
        agent.restore(restore_path)

    #if args.mimic_file is not None:
    #    inject_mimic_experiences(args.mimic_file, buffer)
    run_training(env=env,
                 agent=agent,
                 buffer=buffer,
                 reward_scale=args.reward_scale,
                 batch_size=args.batch_size,
                 num_train_steps=args.num_train_steps,
                 hindsight_agent=False,
                 run_name=args.run_name,
                 render=args.render)
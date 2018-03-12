import numpy as np
import gym

from replay_buffer.replay_buffer import ReplayBuffer
from networks.policy_mixins import AbstractPolicy, MLPPolicy, GaussianPolicy, CategoricalPolicy
from networks.value_function_mixins import MLPValueFunc
from networks.network_interface import AbstractSoftActorCritic

class Agent(GaussianPolicy, MLPPolicy, MLPValueFunc, AbstractPolicy, AbstractSoftActorCritic):
    def __init__(self, s_shape, a_shape):
        super(Agent, self).__init__(s_shape, a_shape)

def run_training(env):
    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    buffer_size = 10**6
    num_train_steps = 4
    batch_size = 32
    reward_scale = 1.0
    s1 = env.reset()

    agent = Agent(state_size, action_size)
    buffer = ReplayBuffer(buffer_size)
    episode_reward = 0
    episodes = 0
    while True:
        a = agent.sample_actions([s1])
        a = a[0]
        s2, r, t, info = env.step(2*a)
        episode_reward += r
        env.render()
        r /= reward_scale
        buffer.append(s1, a, r, s2, t)
        if len(buffer) >= batch_size:
            for i in range(num_train_steps):
                s1_sample, a_sample, r_sample, s2_sample, t_sample = buffer.sample(batch_size)
                [v_loss, q_loss, pi_loss] = agent.train_step(s1_sample, a_sample, r_sample, s2_sample, t_sample)
                print('v_loss', v_loss, 'q_loss', q_loss, 'pi_loss', pi_loss)

        s1 = s2
        if t:
            s1 = env.reset()
            #summary = tf.Summary()
            #summary.value.add(tag='episode reward', simple_value=episode_reward)
            #agent.tb_writer.add_summary(summary, episodes)
            #agent.tb_writer.flush()
            print('Episode %s\t Reward: %s' % (episodes, episode_reward))
            episode_reward = 0
            episodes += 1



#env = gym.make('CartPole-v0')
env = gym.make('Pendulum-v0')
run_training(env)


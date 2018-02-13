import numpy as np
from agents.vector_agent import VectorAgent
from replay_buffer.replay_buffer import ReplayBuffer

def run_training(env):
    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    buffer_size = 10**6
    num_train_steps = 4
    batch_size = 32

    s1 = env.reset()
    agent = VectorAgent(state_size, action_size)
    buffer = ReplayBuffer(state_size, action_size, buffer_size)
    while True:
        a = agent.act(s1)
        s2, r, t, info = env.step(a)
        buffer.append(s1, a, r, s2, t)
        if len(buffer) >= batch_size:
            for i in range(num_train_steps):
                s1_sample, a_sample, r_sample, s2_sample, t_sample = buffer.sample(batch_size)
                [v_loss, q_loss, pi_loss] = agent.train_step(s1_sample, a_sample, r_sample, s2_sample, t_sample)
                print(v_loss, q_loss, pi_loss)
    


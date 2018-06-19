import tensorflow as tf
import numpy as np
import os
from main import build_column_agent
from column_game import ColumnGame
from indep_control2.vae_network import VAE_Network
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from indep_control2.utils import ckpt_surgery

def test():
    nn = VAE_Network(20, 10*10, mode='image')
    nn.restore('./indep_control2/vae_network.ckpt')
    fac_num = 4
    env = ColumnGame(nn, force_max=0.3, reward_per_goal=10.0, indices=[fac_num], visual=False)
    agent = build_column_agent(env, name='SAC2')
    agent.restore(f'./factor_agents/f{fac_num}_copy/weights/sac.ckpt')

def perform_surgery():
    ckpt_surgery('./factor_agents/f4_random/weights/sac.ckpt', lambda x: x, dry_run=True)

#def print_tensors():
#    print_tensors_in_checkpoint_file(f'./factor_agents/f{fac_num}/weights/sac.ckpt', tensor_name='', all_tensors=True)

#test()
perform_surgery()
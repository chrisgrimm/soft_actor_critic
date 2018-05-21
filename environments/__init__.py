from os import path

from gym.envs.registration import register
from os.path import join


def fullpath(dir):
    model_dir = 'models'
    return join(model_dir, dir, 'world.xml')


register(
    id='Navigate-v0',
    entry_point='environments.hsr_gym:HSREnv',
    kwargs={'xml_filepath': fullpath('navigate')})

register(
    id='PickAndPlace-v0',
    entry_point='environments.hsr_gym:HSREnv',
    kwargs={'xml_filepath': fullpath('pick_and_place')})

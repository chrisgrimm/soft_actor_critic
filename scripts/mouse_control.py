#! /usr/bin/env python3
"""Agent that executes random actions"""
# import gym
import argparse
from pathlib import Path

import numpy as np
from click._unicodefun import click

from environments.base import print1
from environments.hindsight_wrapper import PickAndPlaceHindsightWrapper
from environments.pick_and_place import PickAndPlaceEnv
from mujoco import ObjType
import pickle

from sac.utils import Step

saved_pos = None


@click.command()
@click.option('--mimic-path', default=None, type=str)
@click.option('--discrete', is_flag=True)
def cli(discrete, mimic_path):
    # env = NavigateEnv(continuous=True, max_steps=1000, geofence=.5)
    # env = Arm2PosEnv(action_multiplier=.01, history_len=1, continuous=True, max_steps=9999999, neg_reward=True)
    # env = Arm2TouchEnv(action_multiplier=.01, history_len=1, continuous=True, max_steps=9999999, neg_reward=True)
    # env = PickAndPlaceEnv(max_steps=9999999)
    env = PickAndPlaceHindsightWrapper(
        default_reward=0,
        env=PickAndPlaceEnv(random_block=True, discrete=discrete))
    np.set_printoptions(precision=3, linewidth=800)
    env.reset()

    if discrete:
        shape = env.action_space.n
    else:
        shape, = env.action_space.shape

    i = 0
    action = 0 if discrete else np.zeros(shape)
    moving = True
    pause = False
    done = False
    total_reward = 0
    s1 = env.reset()
    traj = []

    while True:
        lastkey = env.env.sim.get_last_key_press()
        if moving:
            if discrete:
                for k in range(1, 7):
                    if lastkey == str(k):
                        action = int(lastkey)

            else:
                action[i] += env.env.sim.get_mouse_dy()

        if lastkey is 'R':
            env.reset()
        if lastkey is ' ':
            moving = not moving
            print('\rmoving:', moving)
        if lastkey is 'P':
            print(env.env.sim.qpos)

        if not discrete:
            for k in range(10):
                if lastkey == str(k):
                    i = k - 1
                    print('')
                    print(env.env.sim.id2name(ObjType.ACTUATOR, i))

        if not pause and not np.allclose(action, 0):
            if not discrete:
                action = np.clip(action, env.action_space.low, env.action_space.high)
            s2, r, done, _ = env.step(action)
            if discrete:
                action = 0


            traj.append(Step(s1, action, r, s2, done))
            s1 = s2
            total_reward += r
            # run_tests(env, s2)

        if done:
            if not pause:
                print('\nresetting', total_reward)
            pause = True
            total_reward = 0
            with Path(mimic_path + '.pkl').open(mode='wb') as f:
                pickle.dump(traj, f)
            exit()
        env.env.render(labels={'x': env.env.goal_3d()})


def run_tests(env, obs):
    assert env.env.observation_space.contains(obs)
    assert not env.env._currently_failed()
    assert np.shape(env._goal()) == np.shape(env.obs_to_goal(obs))
    goal, obs_history = env.destructure_mlp_input(obs)
    assert_equal(env._goal(), goal)
    assert_equal(env._obs(), obs_history[-1])
    assert_equal((goal, obs_history),
                 env.destructure_mlp_input(env.mlp_input(goal, obs_history)))
    assert_equal(obs, env.mlp_input(*env.destructure_mlp_input(obs)))
    assert_equal(obs, env.change_goal(goal, obs))
    try:
        assert_equal(
            env.gripper_pos(), env.gripper_pos(env.sim.qpos), atol=1e-2)
    except AttributeError:
        pass


def assert_equal(val1, val2, atol=1e-5):
    try:
        for a, b in zip(val1, val2):
            assert_equal(a, b, atol=atol)
    except TypeError:
        assert np.allclose(
            val1, val2, atol=atol), "{} vs. {}".format(val1, val2)


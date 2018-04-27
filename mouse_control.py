#! /usr/bin/env python3
"""Agent that executes random actions"""
# import gym
import argparse

import numpy as np
from mujoco import ObjType

from environment.arm2pos import Arm2PosEnv
from environment.arm2touch import Arm2TouchEnv
from environment.base import print1
from environment.navigate import NavigateEnv
from environment.pick_and_place import PickAndPlaceEnv
import pickle

from goal_wrapper import PickAndPlaceGoalWrapper
from sac.main import build_action_converter
from sac.replay_buffer.replay_buffer import ReplayBuffer2

saved_pos = None


def run(port, value_tensor=None, sess=None):
    # env = NavigateEnv(continuous=True, max_steps=1000, geofence=.5)
    #env = Arm2PosEnv(action_multiplier=.01, history_len=1, continuous=True, max_steps=9999999, neg_reward=True)
    # env = Arm2TouchEnv(action_multiplier=.01, history_len=1, continuous=True, max_steps=9999999, neg_reward=True)
    # env = PickAndPlaceEnv(max_steps=9999999)
    env = PickAndPlaceGoalWrapper(PickAndPlaceEnv(max_steps=9999999), ReplayBuffer2(4), 1/10.)
    np.set_printoptions(precision=3, linewidth=800)
    env.reset()
    converter = build_action_converter(env)

    shape, = env.action_space.shape

    i = 0
    j = 0
    action = np.zeros(shape)
    moving = False
    pause = False
    done = False
    total_reward = 0
    s1 = None
    traj = []

    while True:
        lastkey = env.env.sim.get_last_key_press()
        if moving:
            action[i] += env.env.sim.get_mouse_dy()
        # else:
            # for name in ['wrist_roll_motor']:
            # for name in ['slide_x_motor', 'slide_y_motor']:
            # k = env.sim.name2id(ObjType.ACTUATOR, name)
            # action[:] = 0

        if lastkey is 'R':
            env.reset()
        if lastkey is ' ':
            moving = not moving
            print('\rmoving:', moving)
        if lastkey is 'P':
            print(env.env.sim.qpos)

        for k in range(10):
            if lastkey == str(k):
                i = k - 1
                print('')
                print(env.env.sim.id2name(ObjType.ACTUATOR, i))

        # action[1] = .5
        # action *= .05
        if not pause and not np.allclose(action, 0):
            print('.', end='')
            a = np.clip(action * .05, -1, 1)
            s2, r, done, _ = env.step(a, converter)
            print1(r)
            if s1 is not None:
                traj.append((s1, a, r, s2, done))
            s1 = s2
            total_reward += r
            run_tests(env, s2)

        if done:
            if not pause:
                print('\nresetting', total_reward)
            pause = True
            total_reward = 0
            # with open('success_trajectory.pkl', mode='wb') as f:
            #     pickle.dump(traj, f)
        env.env.render(labels={'x': env.env.goal_3d()})


def run_tests(env, obs):
    assert env.env.observation_space.contains(obs)
    assert not env.env._currently_failed()
    assert np.shape(env._goal()) == np.shape(env.obs_to_goal(obs))
    goal, obs_history = env.destructure_mlp_input(obs)
    assert_equal(env._goal(), goal)
    assert_equal(env._obs(), obs_history[-1])
    assert_equal((goal, obs_history), env.destructure_mlp_input(env.mlp_input(goal, obs_history)))
    assert_equal(obs, env.mlp_input(*env.destructure_mlp_input(obs)))
    assert_equal(obs, env.change_goal(goal, obs))
    try:
        assert_equal(env._gripper_pos(), env._gripper_pos(env.sim.qpos), atol=1e-2)
    except AttributeError:
        pass


def assert_equal(val1, val2, atol=1e-5):
    try:
        for a, b in zip(val1, val2):
            assert_equal(a, b, atol=atol)
    except TypeError:
        assert np.allclose(val1, val2, atol=atol), "{} vs. {}".format(val1, val2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=None)
    args = parser.parse_args()

    run(args.port)

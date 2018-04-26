import numpy as np
from hsr_gazebo import HSRGazeboEnv

if __name__ == '__main__':

    env = HSRGazeboEnv(launchfile='/opt/tmc/ros/indigo/share/hsrb_gazebo_launch/launch/hsrb_apartment_no_objects_world_headless.launch')

    # env._reset()

    i = 0

    while True:
        rand_action = np.random.rand(2) - 0.5  # [forward, rotate]
        # rand_action *= 10

        _, _, done, _ = env.step(rand_action)

        if done:
            env._reset()

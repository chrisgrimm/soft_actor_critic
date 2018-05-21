import os
import signal
import subprocess
from os import path

import gym

import rospy
from std_srvs.srv import Empty


class GazeboEnv(gym.Env):
    """Superclass for all Gazebo environments.
    """

    def __init__(self, launchfile):
        # start roscore
        subprocess.Popen("roscore")

        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(
                os.path.dirname(__file__), "assets", "launch", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", fullpath])

        self.gzclient_pid = 0

    def _step(self, action):
        # Perform a step in Gazebo
        raise NotImplementedError

    def _reset(self):
        # Reset environment
        raise NotImplementedError

    def _render(self, close=True):
        pass  # Now done in launch file

        # # Opens Gazebo and shows robot
        # if close:
        #     tmp = os.popen("ps -Af").read()
        #     proccount = tmp.count('gzclient')
        #     if proccount > 0:
        #         if self.gzclient_pid != 0:
        #             os.kill(self.gzclient_pid, signal.SIGTERM)
        #             os.wait()
        #     return

        # tmp = os.popen("ps -Af").read()
        # proccount = tmp.count('gzclient')
        # if proccount < 1:
        #     subprocess.Popen("gzclient")
        #     self.gzclient_pid = int(subprocess.check_output(["pidof","-s","gzclient"]))
        # else:
        #     self.gzclient_pid = 0

    def _close(self):
        # Kill gzclient, gzserver and roscore
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count('gzclient')
        gzserver_count = tmp.count('gzserver')
        roscore_count = tmp.count('roscore')
        rosmaster_count = tmp.count('rosmaster')

        if gzclient_count > 0:
            os.system("killall -9 gzclient")
        if gzserver_count > 0:
            os.system("killall -9 gzserver")
        if rosmaster_count > 0:
            os.system("killall -9 rosmaster")
        # if roscore_count > 0:
        #     os.system("killall -9 roscore")

        if (gzclient_count or gzserver_count or roscore_count
                or rosmaster_count > 0):
            os.wait()

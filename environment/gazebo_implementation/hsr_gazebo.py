"""Create gym environment for HSR inside of Gazebo"""

import os
import pickle
from threading import Lock

import cv2
import rospy
import numpy as np
import scipy.misc

from gazebo_gym_env import GazeboEnv

from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelStates, ModelState
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist, Pose, Quaternion
from rosgraph_msgs.msg import Clock

from cv_bridge import CvBridge, CvBridgeError

from tf.transformations import euler_from_quaternion, quaternion_from_euler

EPSILON = 1e-4
MIN_MOVEMENT_TO_ACTION_RATIO = 0.01
MAX_GOAL_QUERIES = int(1e5)


class HSRGazeboEnv(GazeboEnv):
    """ The environment """

    def __init__(self,
                 image_size=64,
                 use_frame_history=False,
                 action_size=2,
                 history_len=4,
                 max_steps=3000,
                 offscreen=False,
                 geofence_size=1.0,
                 launchfile='/'):

        self._geofence_size = geofence_size
        self._use_frame_history = use_frame_history
        self._frame_buffer = ObsBuffer(history_len, image_size)
        self._image_size = image_size
        self.save_img = False

        self._body_name = 'hsrb'

        self._body_radius = 0.25
        self._max_speed = 20
        self._steps_per_action = 3
        self._max_steps = max_steps
        self._step_num = 0
        self._prev_pos = None
        self._prev_action = None
        self._timesteps_stuck = 20
        self._max_timesteps_stuck = 500

        frame_hist_mult = 4 if use_frame_history else 1

        self.lock = Lock()

        self.excluded_collisions = ['lawn']  # TODO: Check name
        self.objects_initialized = False

        self.robot_pose = [0.0, 0.0, 0.0]  # Pos X, Pos Y, Angular Z

        self.vel_pub = rospy.Publisher(
            '/hsrb/command_velocity', Twist, queue_size=5)
        self.set_state = rospy.Publisher(
            '/gazebo/set_model_state', ModelState, queue_size=5)

        self.unpause_srv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause_srv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy_srv = rospy.ServiceProxy('/gazebo/reset_simulation',
                                                  Empty)

        self.world_lower_bound = np.array([0.0, 0.0])
        self.world_upper_bound = np.array([0.0, 0.0])
        self.bridge = CvBridge()

        GazeboEnv.__init__(self, launchfile=launchfile)

        rospy.init_node('hsrcontrol')
        self.rate = rospy.Rate(1)

        self.ros_first = None
        self.sim_first = None

    def _get_new_pos(self, rel_to=None, bound=None):
        for _ in range(MAX_GOAL_QUERIES):
            if rel_to is None:

                pos = np.random.uniform(
                    self.world_lower_bound + self.world_size * 1 / 5,
                    self.world_upper_bound - self.world_size * 1 / 5)

            else:
                coord, radius = rel_to
                radius = np.random.uniform(self._geofence_size, radius)
                angle = np.random.uniform(0, np.pi)
                offset = np.array([np.cos(angle), np.sin(angle)]) * radius
                pos = coord + offset

            if bound is None:
                bound = 2 * self._body_radius

            bounding_box = np.array([bound, bound])

        return pos

    @property
    def pos_buffer(self):
        if self._use_frame_history:
            xy, img = self._frame_buffer.get()
            return xy
        else:
            return self.pos

    @property
    def pos(self):
        return [self.robot_pose[0], self.robot_pose[1]]  # ignore z-axis

    @property
    def orientation(self):
        return np.array(
            [np.cos(self.robot_pose[2]),
             np.sin(self.robot_pose[2])])

    @property
    def _at_goal(self):
        try:
            distance_to_goal = self.distance_between(self.pos, self.goal)
            return distance_to_goal < self._geofence_size
        except AttributeError:
            return False

    @property
    def normalized_goal(self):
        return (self.goal) / self._world_size

    @property
    def _stuck(self):
        if self._prev_pos is None:
            return False

        frontal_movement = abs(self._prev_action[0])
        if frontal_movement > EPSILON:
            assert self._prev_action is not None
            distance_traveled = self.distance_between(self._prev_pos, self.pos)
            movement_to_action_ratio = distance_traveled / frontal_movement
            if movement_to_action_ratio < MIN_MOVEMENT_TO_ACTION_RATIO:
                self._timesteps_stuck += 1
            else:
                self._timesteps_stuck = 0
            if self._timesteps_stuck > self._max_timesteps_stuck:
                return True
        return False

    @property
    def obs(self):
        msg = rospy.wait_for_message('/hsrb/head_rgbd_sensor/rgb/image_raw',
                                     Image)

        dimensions = msg.height, msg.width
        obs = self.resize_image(msg)

        if self._use_frame_history:
            self._frame_buffer.update(self.pos, obs)
            xy, obs = self._frame_buffer.get()

        return obs

    @property
    def _escaped(self):
        try:
            return np.any(self.pos > self.world_upper_bound) \
                or np.any(self.pos < self.world_lower_bound)
        except AttributeError:
            return False

    def _step(self, action):
        # assert action.shape == self.action_space.shape
        action = np.clip(action, -1, 1)
        self._step_num += 1

        print('Got Action: {}'.format(action))

        step = 0
        reward = 0
        done = False
        while not done and step < self._steps_per_action:
            new_reward, done = self._step_inner(action)
            reward += new_reward
            step += 1
            print('Reward: ', reward, 'Done: ', done)

        print('Reward: ', reward, 'Done: ', done)
        return self.obs, reward, done, {}

    def _step_inner(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_srv()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        msg = None

        while msg is None:
            try:
                msg = rospy.wait_for_message(
                    '/hsrb/base_scan', LaserScan, timeout=5)
            except:
                pass

        self.get_hsrb_state()
        collision = self.check_collisions(msg)

        if self._at_goal:
            print('at goal')
            reward = 1
        elif self._escaped:
            print('escaped')
            reward = -1
        elif self._stuck:
            print('stuck')
            reward = -1
        elif collision:
            print('collision')
            reward = -1
        else:
            reward = 0

        hit_max_steps = self._step_num >= self._max_steps
        done = self._at_goal or hit_max_steps or self._escaped or self._stuck or collision

        if done:
            return done, reward

        self._prev_pos = self.pos
        self._prev_action = action

        # action
        forward, rotate = action  # * self._max_speed
        action = np.append(forward * self.orientation, [rotate])

        reset_cmd = Twist()

        reset_cmd.angular.z = 0
        reset_cmd.linear.x = 0.0
        reset_cmd.linear.y = 0.0

        # if forward < 0:
        #     vel_cmd = Twist()
        #     vel_cmd.angular.z = 180
        #     vel_cmd.linear.x = 0.0
        #     vel_cmd.linear.y = 0.0

        #     forward = abs(forward)
        #     rotate = -1 * (180 - rotate)

        #     self.get_hsrb_state()

        #     while True:
        #         if 180 - abs(old_pose - self.robot_pose[2]) < 1.0:
        #             print('Broke 1, turned around')
        #             break

        #         self.vel_pub.publish(vel_cmd)
        #         self.get_hsrb_state()

        #         self.vel_pub.publish(reset_cmd)

        #     # self.rate.sleep()

        # self.vel_pub.publish(reset_cmd)
        # self.get_hsrb_state()
        # old_pose = self.robot_pose[2]

        self.get_hsrb_state()

        euler = euler_from_quaternion(self.current_q)

        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        yaw += rotate

        new_q = quaternion_from_euler(roll, pitch, yaw)

        vel_cmd = Twist()
        vel_cmd.angular.z = rotate
        vel_cmd.linear.x = 0.0
        vel_cmd.linear.y = 0.0

        while True:
            if abs(new_q.dot(self.current_q)) > 1 - 0.001:
                break

            self.get_hsrb_state()
            self.vel_pub.publish(vel_cmd)

        self.vel_pub.publish(reset_cmd)
        self.get_hsrb_state()

        vel_cmd = Twist()
        vel_cmd.angular.z = 0.0

        stuck_timesteps = 0

        # TODO: Do with timesteps
        while self.distance_between(self.pos, self._prev_pos) < 0.25:
            vel_cmd.linear.y = forward * np.sin(rotate)
            vel_cmd.linear.x = forward * np.cos(rotate)

            self.vel_pub.publish(vel_cmd)

            msg = None

            stuck_timesteps += 1

            while msg is None:
                try:
                    msg = rospy.wait_for_message(
                        '/hsrb/base_scan', LaserScan, timeout=5)
                except:
                    pass

            self.get_hsrb_state()
            collision = self.check_collisions(msg)

            if collision:
                done = True
                break

            if stuck_timesteps > self._max_timesteps_stuck:
                done = True
                break

            self.vel_pub.publish(reset_cmd)

        # self.vel_pub.publish(reset_cmd)
        # self.rate.sleep()
        print(self.robot_pose)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_srv()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        assert done is not None
        return reward, done

    def _reset(self):
        self._step_num = 0
        self.ros_first = None
        self.sim_first = None

        if self._use_frame_history:
            self._frame_buffer.reset()
        self._prev_action, self._prev_pos = None, None

        rospy.wait_for_service('/gazebo/reset_simulation')

        try:
            self.reset_proxy_srv()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        return self.obs

    def initialize_objects(self, msg):
        self.objects = list()
        self.robot_idx = 0

        for i, name in enumerate(msg.name):
            x = msg.pose[i].position.x
            y = msg.pose[i].position.y
            z = msg.pose[i].position.z

            self.objects.append([name, x, y])

            if name == 'hsrb':
                self.robot_idx = i

                self.robot_pose = [x, y, z]

                x = msg.pose[self.robot_idx]._orientation.x
                y = msg.pose[self.robot_idx]._orientation.y
                z = msg.pose[self.robot_idx]._orientation.z
                w = msg.pose[self.robot_idx]._orientation.w

                x, y, self.robot_pose[2] = self.quaternion2euler(w, x, y, z)

            if name in self.excluded_collisions:
                continue

            if x > self.world_upper_bound[0]:
                self.world_upper_bound[0] = x

            if y > self.world_upper_bound[1]:
                self.world_upper_bound[1] = y

            if x < self.world_lower_bound[0]:
                self.world_lower_bound[0] = x

            if y < self.world_lower_bound[1]:
                self.world_lower_bound[1] = y

        self.world_size = self.world_upper_bound - self.world_lower_bound
        self.objects_initialized = True

        self.goal = self._get_new_pos(bound=0)

    def check_collisions(self, msg):
        return self.collisions(msg)

    def collisions(self, msg):
        ranges = np.array(msg.ranges)
        nan_indices = np.isnan(ranges)

        ranges[nan_indices] = 0

        indices = np.linspace(0, len(ranges), num=40)

        for i, val in enumerate(indices):
            if i == 0:
                continue

            ind_tuple = (int(indices[i - 1]), int(val))

            mean = np.mean(ranges[ind_tuple[0]:ind_tuple[1]])

            if mean < 0.15:  # Less than 15 Centimeters mean distance away
                return True

        return False

    def get_hsrb_state(self):
        msg = rospy.wait_for_message(
            '/gazebo/model_states', ModelStates, timeout=5)

        if self.ros_first == None:
            self.ros_first = rospy.Time.now()

            self.sim_first = rospy.wait_for_message('/clock', Clock, timeout=5)
            self.sim_first = rospy.Time(self.sim_first.clock.secs,
                                        self.sim_first.clock.nsecs)

        ros_now = rospy.Time.now() - self.ros_first
        sim_actual = rospy.wait_for_message('/clock', Clock, timeout=5)
        sim_now = rospy.Time(sim_actual.clock.secs,
                             sim_actual.clock.nsecs) - self.sim_first

        rospy.loginfo(
            'Current ROS Time: {}, Current Sim Time: {}, Real time factor: {}'.
            format(ros_now.to_sec(), sim_now.to_sec(),
                   sim_now.to_sec() / ros_now.to_sec()))

        with self.lock:
            if not self.objects_initialized:
                self.initialize_objects(msg)

            else:
                x = msg.pose[self.robot_idx]._orientation.x
                y = msg.pose[self.robot_idx]._orientation.y
                z = msg.pose[self.robot_idx]._orientation.z
                w = msg.pose[self.robot_idx]._orientation.w

                self.current_q = np.array([x, y, z, w])

                x, y, self.robot_pose[2] = self.quaternion2euler(w, x, y, z)

                self.robot_pose[0] = msg.pose[self.robot_idx].position.x
                self.robot_pose[1] = msg.pose[self.robot_idx].position.y

    def quaternion2euler(self, w, x, y, z):
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        euler_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = 1 if t2 > 1 else t2
        t2 = -1 if t2 < -1 else t2
        euler_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        euler_z = np.arctan2(t3, t4)

        return euler_x, euler_y, euler_z

    def distance_between(self, pos1, pos2):
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        return np.sqrt(np.sum(np.square(pos1 - pos2)))

    def resize_image(self, img):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)

        img = np.asarray(cv_image)
        img = scipy.misc.imresize(img, (self._image_size, self._image_size))

        if not self.save_img:
            scipy.misc.imsave('test.png', img)
            self.save_img = True

        return img


class ObsBuffer(object):
    def __init__(self, history_len, image_size):
        self.hl = history_len
        self.image_shape = [image_size, image_size, 3]
        self.reset()

    def update(self, xy, img):
        self.xy_buffer = self.xy_buffer[1:] + [xy]
        self.img_buffer = self.img_buffer[1:] + [img]

    def reset(self):
        self.xy_buffer = [np.zeros(2) for _ in range(self.hl)]
        self.img_buffer = [np.zeros(self.image_shape) for _ in range(self.hl)]

    def get(self):
        assert len(self.xy_buffer) == self.hl
        assert len(self.img_buffer) == self.hl
        xy = np.concatenate(self.xy_buffer, axis=0)
        img = np.concatenate(
            [np.reshape(x, self.image_shape) for x in self.img_buffer], axis=2)
        return xy, img

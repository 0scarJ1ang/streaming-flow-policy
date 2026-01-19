#!/usr/bin/env python3

import rospy
import random
import pickle
from std_msgs.msg import String, Float64
from sensor_msgs.msg import Image, JointState
import numpy as np
from scipy.spatial.transform import Rotation as R
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_input as inputMsg
import json 
import tf

from cv_bridge import CvBridge, CvBridgeError

experiment_number = 0
experiment_subsection = 0


# - Translation: [0.081, -0.018, -0.126]
# - Rotation: in Quaternion [0.012, -0.003, 0.003, 1.000]

# D: [0.13546444475650787, -0.47212305665016174, 8.851876191329211e-05, 0.0010161105310544372, 0.43829241394996643]
# K: [600.8057861328125, 0.0, 325.5308532714844, 0.0, 600.634033203125, 252.90933227539062, 0.0, 0.0, 1.0]
# R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# P: [600.8057861328125, 0.0, 325.5308532714844, 0.0, 0.0, 600.634033203125, 252.90933227539062, 0.0, 0.0, 0.0, 1.0, 0.0]



class PPSLogger():
    def __init__(self, sub_type="Single"):
        self.bridge = CvBridge()
        self.trial = 0
        self.is_recording = False
        self.robot_sub = rospy.Subscriber("/joint_states", JointState, self.state_callback)
        rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg, queue_size=1, callback=self.gripper_callback)
        # self.gripper_sub = rospy.Subscriber("/Robotiq2FGripperRobotInput", inputMsg.Robotiq2FGripper_robot_input, self.gripper_callback)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.states = {}
        self.images = []
        self.gripper_states = {}
        self.image_timestamps = []
        self.ee_pose = {}
        self.cam_pose = {}
        self.listener = tf.TransformListener()

    def state_callback(self, msg):
        if self.is_recording:
            timestamp = str(rospy.Time.now())
            self.states[timestamp] = [msg.velocity, msg.position]
            (trans,rot) = self.listener.lookupTransform('world', 'fake_target', rospy.Time())
            self.ee_pose[timestamp] = (trans,rot)
            (trans_cam,rot_cam) = self.listener.lookupTransform('world', 'camera_color_optical_frame', rospy.Time())
            self.cam_pose[timestamp] = (trans_cam,rot_cam)
            # print(msg.velocity)



    def image_callback(self, msg):
        if self.is_recording:
            timestamp = str(rospy.Time.now())
            image = self.bridge.imgmsg_to_cv2(msg)
            self.image_timestamps.append(timestamp)
            self.images.append(image)


    def gripper_callback(self, msg):
        if self.is_recording:
            timestamp = str(rospy.Time.now())
            self.gripper_states[timestamp] = msg.gPO

    def ros_tower(self):
        print("Initializing ROS tower")
        while not rospy.is_shutdown():
            query = input("Query: ")
            if query == "exit":
                break
            elif 'start' in query:
                self.is_recording = True
                print(f"Beginning recording for session {self.trial}")
            elif query == 'stop':
                ## close files
                self.is_recording = False
                with open(f'/home/clear/oscar_data/robot_{self.trial}.json', 'w') as fp:
                    json.dump(self.states, fp)
                with open(f'/home/clear/oscar_data/gripper_{self.trial}.json', 'w') as fp:
                    json.dump(self.gripper_states, fp)
                with open(f'/home/clear/oscar_data/ee_pose_{self.trial}.json', 'w') as fp:
                    json.dump(self.ee_pose, fp)
                with open(f'/home/clear/oscar_data/cam_pose_{self.trial}.json', 'w') as fp:
                    json.dump(self.cam_pose, fp)

                np.save(f'/home/clear/oscar_data/image_time_{self.trial}', self.image_timestamps)
                np.save(f'/home/clear/oscar_data/images_{self.trial}', self.images)
                print(f"Saved logs for trial {self.trial}")
                self.states = {}
                self.images = []
                self.gripper_states = {}
                self.image_timestamps = []
                self.ee_pose = {}
                self.cam_pose = {}
                self.trial += 1
            else:
                try:
                    self.trial = int(query)
                except:
                    pass




def main():
    rospy.init_node('command', anonymous=True)
    print("asdasdad")
    PPS_log = PPSLogger()
    print("asdasdad")
    PPS_log.ros_tower()
    print("asdasdad")
    rospy.spin()

if __name__ == '__main__':
    main()
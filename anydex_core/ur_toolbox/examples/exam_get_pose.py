import urx
import logging
import time
import numpy as np
import math
import cv2
# from urx import RobotPose
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from ur_toolbox.robot import UR_Camera_Gripper
logging.basicConfig(level=logging.WARN)

GRIPPER_TOTAL_LEN = 0.155
FLANGE_TOTAL_LEN = 0.035
# Robotiq TCP = [0, 0, 0.16, 0, 0, 0]
# Ready Pose = [-0.5, 0, 0.35, 0, -math.pi, 0]

rob = UR_Camera_Gripper("192.168.1.102")
try:
    #rob = urx.Robot("localhost")
    rob.set_tcp((0, 0, GRIPPER_TOTAL_LEN+FLANGE_TOTAL_LEN, 0, 0, 0))

    rob.set_payload(0, (0, 0, 0))
    # rob.movej(rob.readyj)

    # pose = rob.getl()

    # rob.movel(np.array([-0.277, -0.50, 0.36, 2.20809619069388, 2.2256698656036904, 0], dtype = np.float32), acc=1,vel=1)
    pose = rob.getl()
    print('origin pose:{}'.format(pose))





finally:
    rob.close()
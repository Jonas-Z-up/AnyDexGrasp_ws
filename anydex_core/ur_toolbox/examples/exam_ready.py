import urx
import logging
import time
import numpy as np
import math
import cv2

from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from ur_toolbox.robot import UR_Camera_Gripper
from ur_toolbox.transformation.pose import translation_rotation_2_matrix 
logging.basicConfig(level=logging.WARN)

# Robotiq TCP = [0, 0, 0.12, 0, 0, 0] (center definition refer to definition.png in graspnetAPI)
# Ready Pose = [-0.5, 0, 0.35, 0, -math.pi, 0]

rob = UR_Camera_Gripper("192.168.1.102")
try:
    rob.set_tcp((0, 0, 0.12, 0, 0, 0))
    rob.set_payload(0, (0, 0, 0))

    v = 0.05
    a = 0.05

    rob.ready(acc = a, vel=v)

finally:
    rob.close()
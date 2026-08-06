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

    v = 0.01
    a = 0.01
    rotation = np.array([
        [0,1,0],
        [0,0,1],
        [1,0,0]],dtype = np.float32)
    translation = np.array([0.0,0.0,0.3])
    gripper_camera_matrix = translation_rotation_2_matrix(translation,rotation)
    # gripper_camera_matrix = np.array([[0,1,0,0],[0,0,1,0],[1,0,0,0.1],[0,0,0,1]])

    # tcp_base_matrix = rob.get_tcp_base_matrix()
    # print(f'tcp_base_matrix:\n{tcp_base_matrix}')
    # camera_tcp_matrix = rob.get_camera_tcp_matrix()
    # print(f'camera_tcp_matrix:\n{camera_tcp_matrix}')
    # print(f'target gripper_camera_matrix:{gripper_camera_matrix}')
    # target_tcp_base_matrix = rob.gripper_camera_pose_2_tcp_base_pose(gripper_camera_matrix)
    # print(f'tar gripper pose in base coordinate:{rob.get_target_gripper_base_pose(gripper_camera_matrix)}')
    # print(f'execute pose:\n{target_tcp_base_matrix}')
    # rob.execute(target_tcp_base_matrix)
    rob.ready()
    rob.execute_camera_pose(gripper_camera_matrix)
    # rob.gripper_action(200)
    # rob.throw()
    # rob.execute(rob.ready_pose(), acc = a, vel = v)
    # rob.open_gripper()

finally:
    rob.close()
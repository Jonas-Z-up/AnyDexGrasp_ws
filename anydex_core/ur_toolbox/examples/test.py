import urx
import logging
import time
import numpy as np
import math
import cv2

from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from ur_toolbox.robot import UR_Camera_Gripper
from ur_toolbox.transformation.pose import translation_rotation_2_matrix 

def get_robot(robot_ip="192.168.1.102", robot_debug = True):
    robot = UR_Camera_Gripper(robot_ip, camera = None, robot_debug = robot_debug)
    robot.set_tcp((0, 0, 0.12, 0, 0, 0))
    robot.set_payload(0, (0, 0, 0))
    return robot

if __name__ == '__main__':
    v = 0.5
    a = 0.1
    ready_j = [-0.20439654985536748, -1.3991220633136194, 1.023033618927002, -1.192138973866598, -1.5606568495379847, -0.2113125959979456]
    robot = get_robot()
    
    try:

        # robot.movel(robot.ready_pose(), vel = v, acc = a, wait = False)
        # while True:
        #     j = robot.getl()
        #     dist = np.linalg.norm(np.array(j)[:3] - np.array(robot.ready_pose())[:3])
        #     print(dist)
        #     if dist < 0.01:
        #         time.sleep(0.1)
        #         break
        
        # robot.movel(robot.throw_pose(), vel = v, acc = a, wait = False)
        while True:
            robot.movej(robot.readyj, vel = 2.4, acc = 1.8)
            robot.movej(robot.throwj, vel = 2.0, acc = 1.0)
            robot.close_gripper(sleep_time = 0.5)
            robot.open_gripper(sleep_time = 0.5)
            # print('\a')
            # time.sleep(5)
            # robot.movej([-1.506894890462057, -1.6865952650653284, 1.6627697944641113, -1.5380709807025355, -1.56909686723818, -0.008650604878560841])
            # robot.movel(robot.throw_pose(), vel = v, acc = a)
            # print(robot.getj())


    finally:
        robot.close()
import urx
import logging
import time
import numpy as np
import math
import cv2
import os
import pickle
import open3d as o3d
from graspnetAPI import Grasp

from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from ur_toolbox.robot import UR_Camera_Gripper
from ur_toolbox.transformation.pose import translation_rotation_2_matrix, pose_array_2_matrix, matrix_2_translation_rotation

scene = 'scene_clutter_15'
rob = UR_Camera_Gripper("192.168.1.102")
poses = np.array([
    [-0.34167465912391226, -0.23444824679838566, 0.0561410380333472, -1.0726994211091347, 2.4901974382601173, -0.057991201712295375]
])

dump_folder = os.path.join('dump', scene)
if not os.path.exists(dump_folder):
    os.makedirs(dump_folder)
np.save(os.path.join(dump_folder, 'pose.npy'), poses)
try:
    rob.set_tcp((0, 0, 0.12, 0, 0, 0))
    rob.set_payload(0, (0, 0, 0))

    v = 0.15
    a = 0.15
    for idx, pose in enumerate(poses):
        print(f'executing pose {idx}:{pose}')
        rob.ready(acc = a, vel = 1.5*v)
        for i in range(5):
            color_image, depth_image, pcd, colored_depth = rob.get_rgbd_image(return_pcd=True,return_color_depth = True)
            time.sleep(0.1)
        cv2.imwrite(os.path.join(dump_folder,'rgb_%03d.png' % idx), color_image)
        cv2.imwrite(os.path.join(dump_folder,'depth_%03d.png' % idx), depth_image)
        b = (depth_image - 400) / (800 - 400)
        c = (np.stack((b,b,b))*255).astype(np.uint8)
        cv2.imwrite(os.path.join(dump_folder,'depth_vis_%03d.png' % idx), c.transpose((1,2,0)))
        np.save(os.path.join(dump_folder, 'points_%03d.npy' % idx), np.asarray(pcd.points))
        np.save(os.path.join(dump_folder, 'colorss_%03d.npy' % idx), np.asarray(pcd.colors))
        pose_mat = pose_array_2_matrix(pose)
        gripper_camere_pose_mat = rob.tcp_base_pose_2_gripper_camera_pose(pose_mat)
        translation, rotation = matrix_2_translation_rotation(gripper_camere_pose_mat)
        g = Grasp(0, 0.1, 0.02, 0.02, rotation, translation, -1)
        o3d.visualization.draw_geometries([g.to_open3d_geometry(), pcd])
        rob.grasp_and_throw(pose, acc = a, vel = v, approach_dist = 0.1, camera_pose=False, execute_grasp=False)
        # rob.close_gripper()
        # # time.sleep(5)
        # rob.movel(pose + np.array([0,0,0.2,0,0,0]), acc=a, vel=v)
        # rob.throw(acc = a, vel = v)
    rob.ready(acc = a, vel = 2*v)
finally:
    rob.close()
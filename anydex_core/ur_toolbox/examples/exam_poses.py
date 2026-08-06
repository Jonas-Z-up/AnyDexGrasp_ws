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
scene_id = 21
while os.path.exists(os.path.join('dump', 'scene_clutter_%04d' % scene_id)):
    scene_id += 1
scene = 'scene_clutter_%04d' % scene_id
rob = UR_Camera_Gripper("192.168.1.102")

demo = False
poses = np.array([
    [-0.572361985721445, -0.002404737603120297, 0.0755182315445294, 1.680913121088292, 2.238606436225951, 0.5269529928328851],
    [-0.5753268958592115, 0.01848070511654232, 0.04450960887599287, 1.8243146499478444, 1.9048795038025328, 0.3691394002911558],
    [-0.6697358665127248, -0.06048750987234279, -0.03279671735831955, 2.881906009312826, 0.9314541583173988, 0.30961127859066717],
    [-0.67797544278988, -0.05004285975693228, -0.07148379931957287, 1.4317630372656691, 2.601899801766069, 0.09239947465697297],
    [-0.4429382478198255, -0.11254757001325916, -0.03653416134930898, 2.665850177093742, 1.1671471789466965, 0.18872481107111375],
    [-0.6929908161498828, -0.21107394447916009, -0.03615472174015815, -3.054285514453968, 0.26216900978110963, -0.12431977405868605],
    [-0.47128862175273395, -0.2742461784982225, -0.05618520786089043, 1.1425361198775574, 2.8747564063218127, 0.24630816646692402]
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
        rob.execute(pose, acc = a, vel = v, approach_dist = 0.1)
        if not demo:
            rob.close_gripper()
        if demo:
            time.sleep(5)
        rob.movel(pose + np.array([0,0,0.2,0,0,0]), acc=a, vel=v)
        if not demo:
            rob.throw(acc = a, vel = v)
    rob.ready(acc = a, vel = 2*v)
finally:
    rob.close()
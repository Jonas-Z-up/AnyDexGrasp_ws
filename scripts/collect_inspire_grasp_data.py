import copy
import json
import os
import sys
import time
import datetime
import argparse
import torch
import numpy as np
import cv2
import open3d as o3d
import MinkowskiEngine as ME
from graspnetAPI import GraspGroup
from ur_toolbox.robot import UR_Camera_Gripper
try:
    from ur_toolbox.robot.Inspire.InspireHandR_grasp import InspireHandRGraspGroup, grasp_types
except ImportError:
    from ur_toolbox.robot.InspireHandR_grasp import InspireHandRGraspGroup, grasp_types
from multiprocessing import shared_memory
from collections import OrderedDict
import matplotlib.pyplot as plt
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..', 'anydex_core')
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(ROOT_DIR)  # for ur_toolbox
from minkowski_graspnet_single_point import MinkowskiGraspNet
from pt_utils import batch_viewpoint_params_to_matrix
import queue
from itertools import count
from threading import Thread
from queue import Queue
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--save_information_path', default='logs/data/decision_model/inspire/obj140/collect', help='inspire model result information path')
parser.add_argument('--inspire_mesh_json_path', default='/home/jonas/Projects/AnyDexGrasp/generate_mesh_and_pointcloud/inspire_urdf', help='InspireHandR meshes and json path')
parser.add_argument('--grasp_type', default=1, help='the data of DH3 grasp type, 1~8')
parser.add_argument('--robot_ip', required=True, help='Robot IP')
parser.add_argument('--use_graspnet_v2', action='store_true', help='Whether to use graspnet v2 format')
parser.add_argument('--half_views', action='store_true', help='Use only half views in network.')
parser.add_argument('--global_camera', action='store_true', help='Use the settings for global camera.')
cfgs = parser.parse_args()

MAX_GRASP_WIDTH = 0.1
MIN_GRASP_WIDTH = 0.02
BATCH_SIZE = 1
DEBUG = True
CALIB = False
GRIPPER_TOTAL_LEN = 0.155
FLANGE_TOTAL_LEN = 0.055
NUM_OF_INSPIREHAND_DEPTH = 8
NUM_OF_INSPIREHAND_TYPE = 4
INSPIREHANDR_VOXElGRID = 0.002
POINTCLOUD_AUGMENT_NUM = 10


def parse_preds(end_points, use_v2=False):
    ## load preds
    before_generator = end_points['before_generator']  # (B, Ns, 256)
    point_features = end_points['point_features']  # (B, Ns, 512)
    coords = end_points['sinput'].C  # (\Sigma Ni, 4)
    objectness_pred = end_points['stage1_objectness_pred']  # (Sigma Ni, 2)
    objectness_mask = torch.argmax(objectness_pred, dim=1).bool()  # (\Sigma Ni,)
    seed_xyz = end_points['stage2_seed_xyz']  # (B, Ns, 3)
    seed_inds = end_points['stage2_seed_inds']  # (B, Ns)
    grasp_view_xyz = end_points['stage2_view_xyz']  # (B, Ns, 3)
    grasp_view_inds = end_points['stage2_view_inds']
    grasp_view_scores = end_points['stage2_view_scores']
    grasp_scores = end_points['stage3_grasp_scores']  # (B, Ns, A, D)
    grasp_features_two_finger = end_points['stage3_grasp_features'].view(grasp_scores.size()[0], grasp_scores.size()[1], -1) # (B, Ns, 3 + C)
    grasp_widths = MAX_GRASP_WIDTH * end_points['stage3_normalized_grasp_widths']  # (B, Ns, A, D)
    grasp_widths[grasp_widths > MAX_GRASP_WIDTH] = MAX_GRASP_WIDTH

    grasp_preds = []
    grasp_features = []
    grasp_vdistance_list = []
    for i in range(BATCH_SIZE):
        
        cloud_mask_i = (coords[:, 0] == i)
        seed_inds_i = seed_inds[i]
        objectness_mask_i = objectness_mask[cloud_mask_i][seed_inds_i]  # (Ns,)

        if objectness_mask_i.any() == False:
            continue

        seed_xyz_i = seed_xyz[i] # [objectness_mask_i]  # (Ns', 3)
        point_features_i = point_features[i] # [objectness_mask_i]
        
        seed_inds_i = seed_inds_i # [objectness_mask_i]
        before_generator_i = before_generator[i] # [objectness_mask_i]
        grasp_view_xyz_i = grasp_view_xyz[i] # [objectness_mask_i]  # (Ns', 3)
        grasp_view_inds_i = grasp_view_inds[i] # [objectness_mask_i]
        grasp_view_scores_i = grasp_view_scores[i] # [objectness_mask_i]
        grasp_scores_i = grasp_scores[i] # [objectness_mask_i]  # (Ns', A, D)
        grasp_widths_i = grasp_widths[i] # [objectness_mask_i] # (Ns', A, D)
        
        Ns, A, D = grasp_scores_i.size()
        grasp_features_two_finger_i = grasp_features_two_finger[i] # [objectness_mask_i] # (Ns', 3 + C)
        grasp_scores_i_A_D = copy.deepcopy(grasp_scores_i).view(Ns, -1)

        grasp_scores_i = torch.minimum(grasp_scores_i[:,:24,:], grasp_scores_i[:,24:,:])
        seed_inds_i = seed_inds_i.view(Ns, -1)
        grasp_view_inds_i = grasp_view_inds_i.view(Ns, -1)
        grasp_view_scores_i = grasp_view_scores_i.view(Ns, -1)

        grasp_scores_i, grasp_angles_class_i = torch.max(grasp_scores_i, dim=1) # (Ns', D), (Ns', D)
        grasp_angles_i = (grasp_angles_class_i.float()-12) / 24 * np.pi  # (Ns', topk, D)
        # grasp width & vdistance
        grasp_angles_class_i = grasp_angles_class_i.unsqueeze(1) # (Ns', 1, D)
        grasp_widths_pos_i = torch.gather(grasp_widths_i, 1, grasp_angles_class_i).squeeze(1) # (Ns', D)
        grasp_widths_neg_i = torch.gather(grasp_widths_i, 1, grasp_angles_class_i+24).squeeze(1) # (Ns', D)

        ## slice preds by grasp score/depth
        # grasp score & depth
        grasp_scores_i, grasp_depths_class_i = torch.max(grasp_scores_i, dim=1, keepdims=True) # (Ns', 1), (Ns', 1)
        grasp_depths_i = (grasp_depths_class_i.float() + 1) * 0.01  # (Ns'*topk, 1)
        grasp_depths_i -= 0.01
        grasp_depths_i[grasp_depths_class_i==0] = 0.005
        grasp_angles_i = torch.gather(grasp_angles_i, 1, grasp_depths_class_i) # (Ns', 1)
        grasp_widths_pos_i = torch.gather(grasp_widths_pos_i, 1, grasp_depths_class_i) # (Ns', 1)
        grasp_widths_neg_i = torch.gather(grasp_widths_neg_i, 1, grasp_depths_class_i) # (Ns', 1)

        # convert to rotation matrix
        rotation_matrices_i = batch_viewpoint_params_to_matrix(-grasp_view_xyz_i, grasp_angles_i.squeeze(1))
        # # adjust gripper centers
        grasp_widths_i = grasp_widths_pos_i + grasp_widths_neg_i
        rotation_matrices_i = rotation_matrices_i.view(Ns, 9)

        # merge preds
        grasp_preds.append(torch.cat([grasp_scores_i, grasp_widths_i, grasp_depths_i, rotation_matrices_i, seed_xyz_i],axis=1))  # (Ns, 15)
        grasp_features.append(torch.cat([grasp_scores_i_A_D, grasp_features_two_finger_i, before_generator_i, point_features_i, grasp_view_inds_i, grasp_view_scores_i, seed_inds_i, grasp_angles_i*24/np.pi+12, grasp_depths_i], axis=1)) # (Ns'*3, A, D)
        
    return grasp_preds, grasp_features

def get_net(checkpoint_path, use_v2=False):
    if use_v2:
        net = MinkowskiGraspNet(num_depth=5, num_seed=2048, is_training=False, half_views=cfgs.half_views)
    else:
        net = MinkowskiGraspNet(num_depth=4, num_seed=2048, is_training=False, half_views=cfgs.half_views)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    return net

def flip_ggarray(ggarray):
    ggarray_rotations = ggarray[:, 4:13].reshape((-1, 3, 3))
    tcp_x_axis_on_base_frame = ggarray_rotations[:, 1, 1]
    if_flip = [False for _ in range(len(ggarray))]
    for ids, y_x in enumerate(tcp_x_axis_on_base_frame):
        if y_x < 0:
            ggarray_rotations[ids, :3, 0:2] = -ggarray_rotations[ids, :3, 0:2]
            if_flip[ids] = True
    ggarray[:, 4:13] = ggarray_rotations.reshape((-1, 9))
    return ggarray, if_flip

def get_grasp_features(grasp_features_array):
    grasp_features = dict()
    grasp_features['grasp_angles'] = int(grasp_features_array[-3]+0.1)
    grasp_features['grasp_depths'] = int(grasp_features_array[-2]*100+0.1)
    grasp_features['stage3_grasp_scores'] = grasp_features_array[:240].tolist()
    grasp_features['grasp_preds_features'] = grasp_features_array[240:240+480].tolist()
    grasp_features['stage3_grasp_features'] = grasp_features_array[240+480:240+480+512].tolist()
    grasp_features['before_generator'] = grasp_features_array[240+480+512:240+480+512+512].tolist()
    grasp_features['point_features'] = grasp_features_array[240+480+512+512:240+480+512+512+512].tolist()
    grasp_features['point_id'] = int(grasp_features_array[-4])
    grasp_features['if_flip'] = bool(int(grasp_features_array[-1]))
    grasp_features['view_inds'] = bool(int(grasp_features_array[-6]))
    grasp_features['view_score'] = bool(int(grasp_features_array[-5]))
    return grasp_features

def get_robot(robot_ip="192.168.2.102", use_rt=False, robot_debug=True, gripper_type='robotiq',
              gripper_port='/dev/ttyUSB1', global_cam=False):
    robot = UR_Camera_Gripper(robot_ip, use_rt, camera=None, robot_debug=robot_debug, gripper_type=gripper_type,
                              gripper_port=gripper_port, global_cam=global_cam)
    robot.set_tcp((0, 0, 0.044, 0, 0, 0))
    robot.set_payload(1.3, (0, 0, 0.09))
    return robot

def get_depth(existing_shm_depth):
    time.sleep(0.1)
    depths_1 = np.copy(np.ndarray((720, 1280), dtype=np.uint16, buffer=existing_shm_depth.buf))
    return depths_1

def save_grasp_information(two_fingers_ggarray, two_fingers_ggarray_object_ids, InspireHandR_ggarray, 
                            two_fingers_ggarray_source, InspireHandR_ggarray_source, two_fingers_ggarray_object_ids_source,
                            two_fingers_grasp_used, InspireHandR_grasp_used, grasp_features_used, 
                            mat_pose, colors_saved, depths_saved, grasp_features, two_fingers_source_grasp_features,
                            before_collision, after_collision):
    save_path = cfgs.save_information_path

    timeStamp = datetime.datetime.now().timestamp()
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d_%H-%M-%S", timeArray)
    save_path = os.path.join(save_path, grasp_types[str(int(InspireHandR_grasp_used.grasp_type))]['name'], otherStyleTime)
    
    information = OrderedDict()
    two_fingers_ggarray_proposals = []
    InspireHandR_ggarray_proposals = []
    for idx, tfg in enumerate(two_fingers_ggarray):
        two_fingers_ggarray_proposals.append(
            [float(tfg.score), float(tfg.width), float(tfg.height), float(tfg.depth)] +
            np.array(tfg.rotation_matrix).reshape((-1)).tolist() + list(tfg.translation.tolist()) +
            [float(two_fingers_ggarray_object_ids[idx])])
        InspireHandR_ggarray_proposals.append(list(InspireHandR_ggarray[idx].get_array_grasp()))


    two_fingers_ggarray_source_saved = []
    InspireHandR_ggarray_source_saved = []
    for idx, tfg in enumerate(two_fingers_ggarray_source):
        two_fingers_ggarray_source_saved.append(
            [float(tfg.score), float(tfg.width), float(tfg.height), float(tfg.depth)] +
            np.array(tfg.rotation_matrix).reshape((-1)).tolist() + list(tfg.translation.tolist()) +
            [float(two_fingers_ggarray_object_ids_source[idx])])
        InspireHandR_ggarray_source_saved.append(list(InspireHandR_ggarray_source[idx].get_array_grasp()))

    tfg = two_fingers_grasp_used
    two_fingers_array = [float(tfg.score), float(tfg.width), float(tfg.height), float(tfg.depth)] + \
                        np.array(tfg.rotation_matrix).reshape((-1)).tolist() + \
                        list(tfg.translation.reshape(-1).tolist()) + \
                        [float(InspireHandR_grasp_used.object_id)]
    grasp_features_used = get_grasp_features(grasp_features_used)
    information['two_fingers_pose'] = list(two_fingers_array)
    information['InspiredHandR_pose'] = list(InspireHandR_grasp_used.get_array_grasp())
    information['two_fingers_pose_angle_type'] = grasp_features_used['grasp_angles']
    information['two_fingers_pose_depth_type'] = grasp_features_used['grasp_depths']

    information['two_fingers_pose_AD'] = grasp_features_used['stage3_grasp_scores']
    information['grasp_preds_features'] = grasp_features_used['grasp_preds_features']
    information['two_fingers_pose_features'] = grasp_features_used['stage3_grasp_features']
    information['two_fingers_pose_features_before_generator'] = grasp_features_used['before_generator']
    information['point_features'] = grasp_features_used['point_features']
    
    information['point_id'] = grasp_features_used['point_id']
    information['if_flip'] = grasp_features_used['if_flip']
    information['InspiredHandR_pose_finger_type'] = int(InspireHandR_grasp_used.grasp_type + 0.1)
    information['InspiredHandR_pose_depth_type'] = int(InspireHandR_grasp_used.depth*100 + 0.1) - grasp_features_used['grasp_depths']
    information['before_collision'] = before_collision
    information['after_collision'] = after_collision
    information['base_2_tcp1'] = np.array(mat_pose[0]).tolist()
    information['base_2_tcp1_backup'] = np.array(mat_pose[1]).tolist()
    information['tcp_2_gripper'] = np.array(mat_pose[2]).tolist()
    information['base_2_TwoFingersGripper_pose'] = np.array(mat_pose[3]).tolist()
    information['tcp_2_camera'] = np.array(mat_pose[4]).tolist()
    information['base_2_tcp_ready'] = np.array(mat_pose[5]).tolist()

    information['camera_internal'] = [[631.119, 363.884], [919.835, 919.61]]

    restart = False
    print('Is the grasping successful? press 1 successfully, press 2 failed, restart grasping and press 3, exit press 4\n')
    if_success = input('The result is: ')
    while True:
        if if_success == '1':
            information['result'] = True
            break
        elif if_success == '2':
            information['result'] = False
            break
        elif if_success == '3':
                restart = True
                break
        elif if_success == '4':
            exit()
        else:
            if_success = input('Re-enter the result: ')
    if restart:
        return
    information['two_fingers_ggarray_proposals'] = two_fingers_ggarray_proposals
    information['InspireHandR_ggarray_proposals'] = InspireHandR_ggarray_proposals
    information['two_fingers_ggarray_informations_proposals'] = np.array(grasp_features).tolist()

    information['two_fingers_ggarray_source'] = two_fingers_ggarray_source_saved
    information['InspireHandR_ggarray_source_saved'] = InspireHandR_ggarray_source_saved

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'two_fingers_ggarray_informations_source.npy'), two_fingers_source_grasp_features)

    color_path = os.path.join(save_path, 'color.png')
    depth_path = os.path.join(save_path, 'depth.png')
    from PIL import Image
    cv2.imwrite(color_path, (cv2.cvtColor(colors_saved, cv2.COLOR_RGB2BGR) * 255.0).astype(np.float32))
    cv2.imwrite(depth_path, depths_saved)
    json_path = os.path.join(save_path, 'information.json')

    json_file = json.dumps(information, indent=4)
    with open(json_path, 'w') as handle:
        handle.write(json_file)
    print('Saved successfully')

def get_grasp(net, depths, existing_shm_color, voxel_size=0.005):
    fx, fy = 919.835, 919.61
    cx, cy = 631.119, 363.884
    s = 1000.0

    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0.35) & (points_z < 0.68)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    if DEBUG:
        colors = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
        colors = colors[mask].astype(np.float32)

    cloud = None
    if DEBUG:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)

    points = torch.from_numpy(points)
    coords = np.ascontiguousarray(points / voxel_size, dtype=int)
    # Upd Note. API change.
    _, idxs = ME.utils.sparse_quantize(coords, return_index=True)
    coords = coords[idxs]
    points = points[idxs]
    coords_batch, points_batch = ME.utils.sparse_collate([coords], [points])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sinput = ME.SparseTensor(points_batch, coords_batch, device=device)

    end_points = {'sinput': sinput, 'point_clouds': [sinput.F]}
    with torch.no_grad():
        end_points = net(end_points)
        preds, grasp_features = parse_preds(end_points, use_v2=cfgs.use_graspnet_v2)
        if len(preds) == 0:
            print('No grasp detected')
            return None, cloud, points.cuda(), None
        else:
            preds = preds[0]
    # filter
    mask = (preds[:,9] > 0.85) & (preds[:,1] < MAX_GRASP_WIDTH) & (preds[:,1] > MIN_GRASP_WIDTH)
    workspace_mask = (preds[:,12] > -0.25) & (preds[:,12] < 0.25) & (preds[:,13] > -0.20) & (preds[:,13] < 0.05)

    preds = preds[workspace_mask & mask]
    grasp_features = grasp_features[0][workspace_mask & mask]
    if len(preds) == 0:
        print('No grasp detected after masking')
        return None, cloud, points.cuda(), None

    points = points.cuda()
    heights = 0.03 * torch.ones([preds.shape[0], 1]).cuda()
    object_ids = -1 * torch.ones([preds.shape[0], 1]).cuda()
    ggarray = torch.cat([preds[:, 0:2], heights, preds[:, 2:15], preds[:, 15:16], object_ids], axis=-1)

    return ggarray, cloud, points, grasp_features

def load_meshes_pointcloud(path):
    meshes_pcls = dict()
    meshes_pcl_path = os.path.join(path, 'meshes/source_pointclouds/voxel_size_' + str(int(INSPIREHANDR_VOXElGRID * 1000)))
    for type in os.listdir(meshes_pcl_path):
        type_path = os.path.join(meshes_pcl_path, type)
        for name in os.listdir(type_path):
            width = name[:-4]
            name_path = os.path.join(type_path, name)
            meshes_pcl = o3d.io.read_point_cloud(name_path)
            meshes_pcls[type + '_' + width] = meshes_pcl
    return meshes_pcls

def robot_grasp(cfgs):
    net = get_net(cfgs.checkpoint_path, use_v2=cfgs.use_graspnet_v2)
    robot = get_robot(cfgs.robot_ip, robot_debug=True, gripper_type='InspireHandR', global_cam=cfgs.global_camera)
    fail = 0
    existing_shm_color = shared_memory.SharedMemory(name='realsense_color')
    existing_shm_depth = shared_memory.SharedMemory(name='realsense_depth')
    meshes_pcls = load_meshes_pointcloud(cfgs.inspire_mesh_json_path)

    try:
        v = 0.01
        a = 0.01
        if cfgs.global_camera:
            robot.movej(robot.throwj2, acc=a*2,
                        vel=v*3)  # this v and a are anguler, so it should be larger than translational
            t1 = time.time()
            depths = get_depth(existing_shm_depth)
            depths_saved = copy.deepcopy(depths)
            colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
            ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, existing_shm_color)
            t3 = time.time()
            print(f'Net Time:{t3 - t1}')

        else:
            robot.movel(robot.ready_pose(), acc=a*2,
                        vel=v*3)  # this v and a are anguler, so it should be larger than translational

        while True:
            if not cfgs.global_camera:
                t1 = time.time()
                robot.movel(robot.ready_pose(), acc=a * 10, vel=v * 10,
                            wait=True)  # this v and a are anguler, so it should be larger than translational
                time.sleep(0.5)
                print('movel')
                depths = get_depth(existing_shm_depth)
                depths_saved = copy.deepcopy(depths)
                colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
                ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, existing_shm_color)
                t3 = time.time()
                print(f'Net Time:{t3 - t1}')

            if ggarray is None:
                fail = fail + 1
                if not cfgs.global_camera:
                    while robot.is_program_running():
                        robot.stopj(acc=10.0 * a)
                    robot.movel(robot.ready_pose(), acc=a*10,
                                vel=v*10)  # this v and a are anguler, so it should be larger than translational
                    time.sleep(0.1)
                else:
                    depths = get_depth(existing_shm_depth)
                    ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, existing_shm_color)
                    time.sleep(0.1)
                if DEBUG:
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                    o3d.visualization.draw_geometries([cloud, frame, sphere])
                continue

            ########## PROCESS GRASPS ##########
            # collision detection
            ggarray = ggarray.cpu().numpy()
            source_index = ggarray[:, 0].argsort()
            ggarray = ggarray[source_index][::-1][:500]

            # Prevent the robot arm from crossing the border, 
            ggarray, if_flip = flip_ggarray(ggarray)

            grasp_features = grasp_features.cpu().numpy()
            two_fingers_source_grasp_features = copy.deepcopy(grasp_features)
            grasp_features = grasp_features[source_index][::-1][:500]
            grasp_features = np.c_[grasp_features, if_flip]
            assert cfgs.grasp_type >=1 and cfgs.grasp_type <= 8
            InspireHandR_types = np.random.randint(cfgs.grasp_type, cfgs.grasp_type + 1, (len(ggarray),))
            InspireHandR_depths = (np.random.randint(0, 4, (len(ggarray),))) * 0.01
            for ids, grasp_type in enumerate(InspireHandR_types):
                if grasp_type == 5:
                    InspireHandR_depths[ids] = InspireHandR_depths[ids] + 0.02 


            two_fingers_ggarray = GraspGroup(ggarray)
            two_fingers_ggarray_object_ids_source = ggarray[:, 16]
            two_fingers_ggarray_source = copy.deepcopy(two_fingers_ggarray)

            InspireHandR_ggarray = InspireHandRGraspGroup() 
            InspireHandR_ggarray.set_grasp_min_width(MIN_GRASP_WIDTH)
            InspireHandR_ggarray.from_graspgroup(two_fingers_ggarray, InspireHandR_types, cfgs.inspire_mesh_json_path)
            InspireHandR_ggarray.object_ids = two_fingers_ggarray_object_ids_source
            InspireHandR_ggarray.depths = InspireHandR_ggarray.depths + InspireHandR_depths
            InspireHandR_ggarray_source = copy.deepcopy(InspireHandR_ggarray)

            index_filter_by_z_axis = InspireHandR_ggarray.filter_grasp_group_by_z_axis(0.4)
            two_fingers_ggarray = two_fingers_ggarray[index_filter_by_z_axis]
            two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids_source[index_filter_by_z_axis]
            grasp_features = grasp_features[index_filter_by_z_axis]

            before_collision = [copy.deepcopy(np.array(InspireHandR_ggarray.grasp_group_array).tolist()), 
                                copy.deepcopy(np.array(two_fingers_ggarray.grasp_group_array).tolist()),
                                copy.deepcopy(np.array(grasp_features).tolist())]
            if len(InspireHandR_ggarray) == 0:
                print('No grasp detected after filter')
                if cfgs.global_camera:
                    ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, existing_shm_color)
                if DEBUG:
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                    o3d.visualization.draw_geometries([cloud, frame, sphere])
                continue

            start_time = time.time()
            approach_distance = 0.04
            mfcdetector = ModelFreeCollisionDetectorInspireHandR(points_down.cpu().numpy(), voxel_size=0.001)
            InspireHandR_ggarray, two_fingers_ggarray, empty_mask, min_width_index = mfcdetector.detect(InspireHandR_ggarray, two_fingers_ggarray,
                                                                  cfgs.inspire_mesh_json_path, meshes_pcls, min_grasp_width=MIN_GRASP_WIDTH,
                                                                  VoxelGrid=INSPIREHANDR_VOXElGRID, DEBUG=False, approach_dist=approach_distance,
                                                                  collision_thresh=1, adjust_gripper_centers=False)

            # proposals
            InspireHandR_ggarray = InspireHandR_ggarray[empty_mask]
            two_fingers_ggarray = two_fingers_ggarray[empty_mask]
            two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[min_width_index][empty_mask]
            grasp_features = grasp_features[min_width_index][empty_mask]

            after_collision = [copy.deepcopy(np.array(InspireHandR_ggarray.grasp_group_array).tolist()), 
                                copy.deepcopy(np.array(two_fingers_ggarray.grasp_group_array).tolist()),
                                copy.deepcopy(np.array(grasp_features).tolist())]


            if len(InspireHandR_ggarray) == 0:
                print('No Grasp detected after collision detection!')
                fail = fail + 1
                if DEBUG:
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                    o3d.visualization.draw_geometries([cloud, frame, sphere])
                if not cfgs.global_camera:
                    while robot.is_program_running():
                        robot.stopj(acc=10.0 * a)
                    robot.movel(robot.ready_pose(), acc=a*10,
                                vel=v*10)  # this v and a are anguler, so it should be larger than translational
                else:
                    t1 = time.time()
                    depths = get_depth(existing_shm_depth)
                    depths_saved = copy.deepcopy(depths)
                    colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
                    ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, existing_shm_color)
                    t3 = time.time()
                    print(f'Net Time:{t3 - t1}')
                continue

            # sort
            index_score = np.argsort(InspireHandR_ggarray.scores)[::-1]
            two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_score][0:10]
            InspireHandR_ggarray = InspireHandR_ggarray[index_score][0:10]
            two_fingers_ggarray = two_fingers_ggarray[index_score][0:10]
            grasp_features = grasp_features[index_score][0:10]

            InspireHandR_grasp_used = InspireHandR_ggarray[0]
            two_fingers_grasp_used = two_fingers_ggarray[0]
            grasp_features_used = grasp_features[0]


            print('picked by scores rotations, translations: ', InspireHandR_grasp_used.rotation_matrix,
                  InspireHandR_grasp_used.translation, two_fingers_grasp_used.translation, two_fingers_grasp_used.rotation_matrix)
            print('grasp score:', InspireHandR_grasp_used.score, two_fingers_grasp_used.score)
            print('grasp width:', InspireHandR_grasp_used.width, two_fingers_grasp_used.width)
            print('grasp depth:', InspireHandR_grasp_used.depth, two_fingers_grasp_used.depth)
            print('grasp type:', InspireHandR_grasp_used.grasp_type)
            print('grasp angle:', InspireHandR_grasp_used.angle)
            t4 = time.time()

            ####################################
            if DEBUG:
                InspireHandR_pose = InspireHandR_grasp_used.load_mesh(cfgs.inspire_mesh_json_path, two_fingers_grasp_used)
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                InspireHandR_pose.paint_uniform_color([1, 0, 0])
                meshes_pointclouds = InspireHandR_grasp_used.load_mesh_pointclouds(cfgs.inspire_mesh_json_path, two_fingers_grasp_used, voxel_size=0.002)
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=meshes_pointclouds,
                                                                            voxel_size=0.002)
                scene_cloud = o3d.geometry.PointCloud()
                scene_cloud.points = o3d.utility.Vector3dVector(points_down.cpu().numpy())
                scene_cloud = scene_cloud.voxel_down_sample(0.001)
                ps = scene_cloud.points
                ps = o3d.utility.Vector3dVector(ps)
                output = voxel_grid.check_if_included(ps)
                o3d.visualization.draw_geometries(
                    [InspireHandR_pose, cloud, sphere, frame, two_fingers_grasp_used.to_open3d_geometry()])
            gripper_time = 0.5
            print('angle: ', InspireHandR_grasp_used.angle)
            robot.open_gripper(InspireHandR_grasp_used.angle)
            mat_pose = robot.grasp_and_throw(InspireHandR_grasp_used, two_fingers_grasp_used, cloud, cfgs.inspire_mesh_json_path,
                                             acc=a*2, vel=v*3, approach_dist=approach_distance,
                                             execute_grasp=True, use_ready_pose=True, gripper_time=gripper_time)
            while robot.is_program_running():
                pass

            t45 = time.time()
                

            save_grasp_information(two_fingers_ggarray, two_fingers_ggarray_object_ids, InspireHandR_ggarray, 
                            two_fingers_ggarray_source, InspireHandR_ggarray_source, two_fingers_ggarray_object_ids_source,
                            two_fingers_grasp_used, InspireHandR_grasp_used, grasp_features_used, 
                            mat_pose, colors_saved, depths_saved, grasp_features, two_fingers_source_grasp_features,
                            before_collision, after_collision)
            if cfgs.global_camera:
                t1 = time.time()
                depths = get_depth(existing_shm_depth)
                depths_saved = copy.deepcopy(depths)
                colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
                ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, existing_shm_color)
                t3 = time.time()
                print(f'Net Time:{t3 - t1}')

            t5 = time.time()
            print(f'ready time:{t5 - t45}')
            print(f'Exec Time:{t5 - t4}')
            mpph = 3600 / (t5 - t1)
            print(f'\033[1;31mMPPH:{mpph}\033[0m\n--------------------')
    finally:
        robot.close()
        existing_shm_depth.close()
        if DEBUG:
            existing_shm_color.close()


if __name__ == '__main__':

    t0 = time.time()
    try:
        robot_grasp(cfgs)
    finally:
        tn = time.time()
        print(f'total time:{tn - t0}')
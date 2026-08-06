import copy
import json
import os
import sys
import time
import datetime
import random
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
from models.minkowski_graspnet_single_point import MinkowskiGraspNet, MinkowskiGraspNetMultifingerType1Inference
from np_utils import transform_point_cloud
from pt_utils import batch_viewpoint_params_to_matrix
from collision_detector import ModelFreeCollisionDetectorMultifinger, ModelFreeCollisionDetectorMultifinger
import queue
from itertools import count
from threading import Thread
from queue import Queue
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--inspire_model_path', default='logs/model/inspire_model/obj140', help='inspire model checkpoint path')
parser.add_argument('--save_information_path', default='logs/data/inspire/inspire_test/obj140', help='inspire model result information path')
parser.add_argument('--inspire_mesh_json_path', default='/home/jonas/Projects/AnyDexGrasp/generate_mesh_and_pointcloud/inspire_urdf', help='InspireHandR meshes and json path')
parser.add_argument('--robot_ip', required=True, help='Robot IP')
parser.add_argument('--use_graspnet_v2', action='store_true', help='Whether to use graspnet v2 format')
parser.add_argument('--half_views', action='store_true', help='Use only half views in network.')
parser.add_argument('--global_camera', action='store_true', help='Use the settings for global camera.')
cfgs = parser.parse_args()

MAX_GRASP_WIDTH = 0.1
MIN_GRASP_WIDTH = 0.01
BATCH_SIZE = 1
DEBUG = False
CALIB = False
GRIPPER_TOTAL_LEN = 0.155
FLANGE_TOTAL_LEN = 0.055
INSPIREHANDR_DEFAULT_DEPTH = 0.000
NUM_OF_INSPIRE_DEPTH = 4
NUM_OF_INSPIRE_TYPE = 8
INSPIREHANDR_VOXElGRID = 0.003
POINTCLOUD_AUGMENT_NUM = 10
RANDOM_GRASP = True

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
        # grasp angle & width & vdistance
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

def get_graspgroup_features(grasp_features_array, sinput):
    grasp_features = dict()
    grasp_features['grasp_angles'] = (grasp_features_array[:, -3]+0.1).astype(int)
    grasp_features['grasp_depths'] = (grasp_features_array[:, -2]*100+0.1).astype(int)
    grasp_features['stage3_grasp_scores'] = grasp_features_array[:, :240]
    grasp_features['grasp_preds_features'] = grasp_features_array[:, 240:240+480]
    grasp_features['stage3_grasp_features'] = grasp_features_array[:, 240+480:240+480+512]
    grasp_features['before_generator'] = grasp_features_array[:, 240+480+512:240+480+512+512]
    grasp_features['point_features'] = grasp_features_array[:, 240+480+512+512:240+480+512+512+512]
    grasp_features['point_id'] = (grasp_features_array[:, -4]+0.1).astype(int)
    grasp_features['if_flip'] = grasp_features_array[:, -1]
    grasp_features['view_inds'] = (grasp_features_array[:, -6]+0.1).astype(int)
    grasp_features['view_score'] = grasp_features_array[:, -5]
    grasp_features['sinput'] = sinput

    grasp_preds_features_rot = np.zeros(grasp_features['grasp_preds_features'].shape, dtype = np.float32)
    new_type = grasp_features['grasp_angles']
    for idx, if_flip in enumerate(grasp_features['if_flip']):
        if if_flip:
            first_half_scores = copy.deepcopy(grasp_features['grasp_preds_features'][idx, :120])
            last_half_scores = copy.deepcopy(grasp_features['grasp_preds_features'][idx, 120:240])
            first_half_widths = copy.deepcopy(grasp_features['grasp_preds_features'][idx, 240:360])
            last_half_widths = copy.deepcopy(grasp_features['grasp_preds_features'][idx, 360:480])
            grasp_features['grasp_preds_features'][idx, :120] = last_half_scores
            grasp_features['grasp_preds_features'][idx, 120:240] = first_half_scores
            grasp_features['grasp_preds_features'][idx, 240:360] = last_half_widths
            grasp_features['grasp_preds_features'][idx, 360:480] = first_half_widths
        grasp_preds_features_rot[idx, :240-new_type[idx]*5] = grasp_features['grasp_preds_features'][idx, new_type[idx]*5:240]
        grasp_preds_features_rot[idx, 240-new_type[idx]*5:240] = grasp_features['grasp_preds_features'][idx, 0:new_type[idx]*5]
        grasp_preds_features_rot[idx, 240:480-new_type[idx]*5] = grasp_features['grasp_preds_features'][idx, 240+new_type[idx]*5:480]
        grasp_preds_features_rot[idx, 480-new_type[idx]*5:480] = grasp_features['grasp_preds_features'][idx, 240:240+new_type[idx]*5] 
    grasp_features['grasp_preds_features'] = grasp_preds_features_rot

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
                            two_fingers_grasp_used, InspireHandR_grasp_used, grasp_informations_used, 
                            mat_pose, colors_saved, depths_saved, grasp_informations):
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
    information['two_fingers_pose'] = list(two_fingers_array)
    information['InspiredHandR_pose'] = list(InspireHandR_grasp_used.get_array_grasp())
    information['two_fingers_pose_angle_type'] = int(grasp_informations_used[0] + 0.1)
    information['two_fingers_pose_depth_type'] = int(grasp_informations_used[1]*100 + 0.1)

    information['two_fingers_pose_depth_type'] = grasp_features_used['grasp_depths']
    information['point_id'] = grasp_features_used['point_id']
    information['if_flip'] = grasp_features_used['if_flip']
    information['InspiredHandR_pose_finger_type'] = int(InspireHandR_grasp_used.grasp_type + 0.1)
    information['InspiredHandR_pose_depth_type'] = int(InspireHandR_grasp_used.depth*100 + 0.1) - grasp_features_used['grasp_depths']

    information['two_fingers_pose_AD'] = grasp_features_used['stage3_grasp_scores']
    information['two_fingers_pose_features_grasp_preds'] = grasp_features_used['grasp_preds_features']
    information['two_fingers_pose_feature_and_AD'] = np.array(grasp_informations_used[2:]).tolist()
    information['two_fingers_pose_features'] = grasp_features_used['stage3_grasp_features']
    information['two_fingers_pose_features_before_generator'] = grasp_features_used['before_generator']
    information['point_features'] = grasp_features_used['point_features']
    
    information['base_2_tcp1'] = np.array(mat_pose[0]).tolist()
    information['base_2_tcp1_backup'] = np.array(mat_pose[1]).tolist()
    information['tcp_2_gripper'] = np.array(mat_pose[2]).tolist()
    information['base_2_TwoFingersGripper_pose'] = np.array(mat_pose[3]).tolist()
    information['tcp_2_camera'] = np.array(mat_pose[4]).tolist()
    information['base_2_tcp_ready'] = np.array(mat_pose[5]).tolist()

    information['camera_internal'] = [[629.535, 351.636], [912.897, 912.258]]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

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

def get_grasp(net, depths, existing_shm_color, augment_mat=np.eye(4), flip=False, voxel_size=0.005,
              min_depth=0.35, max_depth=0.68, intrinsics=None,
              ws_x_range=(-0.40, 0.40), ws_y_range=(-0.40, 0.40),
              preprocess=False,
              cloud_ws_x_range=(-0.5, 0.5), cloud_ws_y_range=(-0.5, 0.5),
              remove_table=False, denoise=False,
              plane_dist_thresh=0.008, table_clearance=0.01,
              plane_min_inliers_ratio=0.15, ransac_max_iter=100, plane_sample=50000,
              sor_nb_neighbors=20, sor_std_ratio=2.0, sor_max_points=200000,
              use_frame_cache=True):
    if intrinsics is None:
        fx, fy = 919.835, 919.61
        cx, cy = 631.119, 363.884
    else:
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['ppx'], intrinsics['ppy']
    s = 1000.0

    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > min_depth) & (points_z < max_depth)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    print(f'[get_grasp] masked points: {points.shape[0]}, voxel_size: {voxel_size}', flush=True)
    if points.shape[0] == 0:
        print(f'[get_grasp] no points in depth range {min_depth:.3f}-{max_depth:.3f}m', flush=True)
        return None, None, torch.empty((0, 3), device='cuda' if torch.cuda.is_available() else 'cpu'), None, None

    # ── optional preprocessing ─────────────────────────────────
    keep_mask = np.ones(points.shape[0], dtype=bool)
    if preprocess:
        from pointcloud_preprocess import preprocess_point_cloud_cached
        cfg = dict(
            crop_xy=True, cloud_ws_x_range=cloud_ws_x_range, cloud_ws_y_range=cloud_ws_y_range,
            remove_table=remove_table, denoise=denoise,
            plane_dist_thresh=plane_dist_thresh, table_clearance=table_clearance,
            plane_min_inliers_ratio=plane_min_inliers_ratio,
            ransac_max_iter=ransac_max_iter, plane_sample=plane_sample,
            sor_nb_neighbors=sor_nb_neighbors, sor_std_ratio=sor_std_ratio,
            sor_max_points=sor_max_points,
        )
        if use_frame_cache:
            points, keep_mask = preprocess_point_cloud_cached(points, cfg, depths)
        else:
            from pointcloud_preprocess import preprocess_point_cloud
            points, keep_mask = preprocess_point_cloud(points, cfg)
        print(f'[get_grasp] after preprocess: {points.shape[0]} points', flush=True)
        if points.shape[0] < 256:
            print('[get_grasp] too few points after preprocessing', flush=True)
            return None, None, torch.empty((0, 3), device='cuda' if torch.cuda.is_available() else 'cpu'), None, None

    if DEBUG:
        colors = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
        colors = colors[mask].astype(np.float32)
        if preprocess:
            colors = colors[keep_mask]

    cloud = None
    if DEBUG:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)

    points = transform_point_cloud(points, augment_mat).astype(np.float32)
    points = torch.from_numpy(points)
    coords = np.ascontiguousarray(points / voxel_size, dtype=int)
    # Upd Note. API change.
    _, idxs = ME.utils.sparse_quantize(coords, return_index=True)
    coords = coords[idxs]
    points = points[idxs]
    coords_batch, points_batch = ME.utils.sparse_collate([coords], [points])
    print(f'[get_grasp] sparse points: {points.shape[0]}', flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sinput = ME.SparseTensor(points_batch, coords_batch, device=device)
    print('[get_grasp] sparse tensor built', flush=True)

    end_points = {'sinput': sinput, 'point_clouds': [sinput.F]}
    with torch.no_grad():
        print('[get_grasp] network forward start', flush=True)
        end_points = net(end_points)
        print('[get_grasp] network forward done', flush=True)
        preds, grasp_features = parse_preds(end_points, use_v2=cfgs.use_graspnet_v2)
        print('[get_grasp] parse preds done', flush=True)
        if len(preds) == 0:
            print('No grasp detected')
            return None, cloud, points.cuda(), None, None
        else:
            preds = preds[0]
    # filter
    if flip:
        augment_mat[:, 0] = -augment_mat[:, 0]
    augment_mat_tensor = torch.tensor(copy.deepcopy(np.linalg.inv(augment_mat).astype(np.float32)), device=device)
    rotation = augment_mat_tensor[:3, :3].reshape((-1)).repeat((preds.size()[0], 1)).view((preds.size()[0], 3, 3))
    translation = augment_mat_tensor[:3, 3]

    preds[:,12:15] = torch.matmul(rotation, preds[:,12:15].view((-1, 3, 1))).view(-1, 3) + translation
    pose_rotation = torch.matmul(rotation, preds[:,3:12].view((-1, 3, 3)))
    if flip:
        preds[:, 12] = -preds[:, 12]
        pose_rotation[:, 0, :] = -pose_rotation[:, 0, :]
        pose_rotation[:, :, 1] = -pose_rotation[:, :, 1]
    preds[:, 3:12] = pose_rotation.view((-1, 9))

    total_candidates = len(preds)
    score_mask = preds[:,9] > 0.92
    width_mask = (preds[:,1] < MAX_GRASP_WIDTH) & (preds[:,1] > MIN_GRASP_WIDTH)
    ws_x_mask = (preds[:,12] > ws_x_range[0]) & (preds[:,12] < ws_x_range[1])
    ws_y_mask = (preds[:,13] > ws_y_range[0]) & (preds[:,13] < ws_y_range[1])
    quality_pass = (score_mask & width_mask).sum().item()
    ws_pass = (ws_x_mask & ws_y_mask).sum().item()
    combined = (score_mask & width_mask & ws_x_mask & ws_y_mask).sum().item()
    print(f'[get_grasp] candidates: {total_candidates}  '
          f'quality(score&width): {quality_pass}  '
          f'workspace(x&y): {ws_pass}  '
          f'combined: {combined}',
          flush=True)
    if combined == 0 and quality_pass > 0 and ws_pass > 0:
        # 检查是否是两群不重叠
        q_idx = (score_mask & width_mask).nonzero(as_tuple=True)[0]
        w_idx = (ws_x_mask & ws_y_mask).nonzero(as_tuple=True)[0]
        q_set, w_set = set(q_idx.tolist()), set(w_idx.tolist())
        overlap = len(q_set & w_set)
        print(f'[get_grasp] quality∩workspace: {overlap}  '
              f'(quality独有的: {len(q_set - w_set)}, '
              f'workspace独有的: {len(w_set - q_set)})',
              flush=True)
    mask = score_mask & width_mask
    workspace_mask = ws_x_mask & ws_y_mask
    preds = preds[workspace_mask & mask]
    grasp_features = grasp_features[0][workspace_mask & mask]
    if len(preds) == 0:
        print('No grasp detected after masking')
        return None, cloud, points.cuda(), None, None

    points = points.cuda()
    heights = 0.03 * torch.ones([preds.shape[0], 1]).cuda()
    object_ids = -1 * torch.ones([preds.shape[0], 1]).cuda()
    ggarray = torch.cat([preds[:, 0:2], heights, preds[:, 2:15], preds[:, 15:16], object_ids], axis=-1)

    return ggarray, cloud, points, grasp_features, [sinput]

def load_meshes_pointcloud(path):
    meshes_pcls = dict()
    meshes_pcl_path = os.path.join(path, 'source_pointclouds/voxel_size_' + str(int(INSPIREHANDR_VOXElGRID * 1000)))
    if not os.path.isdir(meshes_pcl_path):
        meshes_pcl_path = os.path.join(path, 'meshes/source_pointclouds/voxel_size_' + str(int(INSPIREHANDR_VOXElGRID * 1000)))
    for type in os.listdir(meshes_pcl_path):
        type_path = os.path.join(meshes_pcl_path, type)
        for name in os.listdir(type_path):
            width = name[:-4]
            name_path = os.path.join(type_path, name)
            meshes_pcl = o3d.io.read_point_cloud(name_path)
            meshes_pcls[type + '_' + width] = meshes_pcl
    return meshes_pcls

def get_inspire_model(inspire_models_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inspire_models = dict()
    for model_type in os.listdir(inspire_models_path):
        inspire_model_path = os.path.join(inspire_models_path, model_type)
        inspire_models[model_type] = dict()
        for model_class in os.listdir(inspire_model_path):
            model_classs_type = os.path.join(inspire_model_path, model_class)
            for model in os.listdir(model_classs_type):
                inspire_model_type_path = os.path.join(model_classs_type, model)
                inspire_model = MinkowskiGraspNetMultifingerType1Inference(input_num=int(model_type))
                inspire_net = torch.load(inspire_model_type_path)
                inspire_model.load_state_dict(inspire_net.state_dict())
                inspire_model.to(device)
                inspire_model.eval()
                inspire_models[model_type][model_class] = inspire_model
    return inspire_models

def get_inspire_depth_type(inspire_models, grasp_features_dic, ggarray, grasp_features):
    if RANDOM_GRASP:
        inspire_type = np.random.randint(0, NUM_OF_INSPIRE_TYPE, (len(ggarray),))
        inspire_depth = (np.random.randint(0, 4, (len(ggarray),)))
        inspire_depth[inspire_type==5] = inspire_depth[inspire_type==5] + 2
        inspire_depth = inspire_depth * 0.01
        scores = ggarray[:, 0] 
        return inspire_depth, inspire_type, scores, ggarray, grasp_features
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for k, v in grasp_features_dic.items():
        if k == 'point_id' or k == 'sinput':
            continue
        grasp_features_dic[k] = torch.tensor(copy.deepcopy(v), device=device)

    inspire_depth_type_scores = []
    for model_type, inspire_model in inspire_models.items():
        if model_type == '240':
            continue
            model_input = torch.cat([grasp_features_dic["stage3_grasp_scores"]], dim=1)
        elif model_type == '480':
            print('use final model: ', model_type)
            model_input = torch.cat([grasp_features_dic["grasp_preds_features"]], dim=1)
        

        inspire_model_sorted = OrderedDict(sorted(inspire_model.items(), key = lambda t : int(t[0])))
        for model_class, sub_inspire_model in inspire_model_sorted.items():
            with torch.no_grad():
                grasp_pred, _ = sub_inspire_model(model_input) # (B, 1， NUM_OF_TWO_FINGER_DEPTH*NUM_OF_INSPIRE_DEPTH)
                grasp_pred = grasp_pred.view(grasp_pred.shape[0], 5*NUM_OF_INSPIRE_DEPTH)
            two_fingers_depth = grasp_features_dic['grasp_depths'] # (B, )
            base = torch.tensor(np.array([[i for i in range(NUM_OF_INSPIRE_DEPTH)]
                                  for _ in range(grasp_pred.size()[0])]), device=device) # (B, NUM_OF_INSPIRE_DEPTH)
            select_index = (two_fingers_depth).view(-1, 1) * NUM_OF_INSPIRE_DEPTH + base
            inspire_depth_type_scores.append(grasp_pred.gather(1, select_index))
    inspire_depth_type_scores = torch.cat(inspire_depth_type_scores, axis=1).view(-1) # (B, NUM_OF_INSPIRE_DEPTH*NUM_OF_INSPIRE_TYPE)
    scores, index = inspire_depth_type_scores.topk(min(3000, inspire_depth_type_scores.size()[0]))
    pose_index = (index / (NUM_OF_INSPIRE_DEPTH * NUM_OF_INSPIRE_TYPE)).long()
    ggarray = torch.tensor(copy.deepcopy(ggarray), device=device)[pose_index]
    grasp_features = torch.tensor(copy.deepcopy(grasp_features), device=device)[pose_index]
    inspire_depth = ((index % (NUM_OF_INSPIRE_DEPTH * NUM_OF_INSPIRE_TYPE)) % NUM_OF_INSPIRE_DEPTH).int()
    inspire_type = ((index % (NUM_OF_INSPIRE_DEPTH * NUM_OF_INSPIRE_TYPE)) / NUM_OF_INSPIRE_DEPTH).int()

    inspire_depth[inspire_type==5] = inspire_depth[inspire_type==5] + 2
    inspire_depth = inspire_depth * 0.01

    return inspire_depth.cpu().numpy(), inspire_type.cpu().numpy() + 1, scores.detach().cpu().numpy(), \
                ggarray.cpu().numpy(), grasp_features.cpu().numpy()

def augment_data(flip=False):
    flip_mat = np.identity(4)
    # Flipping along the YZ plane
    if flip:
        flip_mat = np.array([[-1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

    # Rotation along up-axis/Z-axis
    rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    rot_mat = np.array([[c, -s, 0, 0],
                        [s, c, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    # Translation along X/Y/Z-axis
    offset_x = np.random.random() * 0.1 - 0.05  # -0.05 ~ 0.05
    offset_y = np.random.random() * 0.1 - 0.05  # -0.05 ~ 0.05
    trans_mat = np.array([[1, 0, 0, offset_x],
                          [0, 1, 0, offset_y],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

    aug_mat = np.dot(trans_mat, np.dot(rot_mat, flip_mat).astype(np.float32)).astype(np.float32)
    return aug_mat

def get_ggarray_features(existing_shm_depth, existing_shm_color, net):
    depths = get_depth(existing_shm_depth)
    augment_mat1 = np.eye(4)
    augment_mats = []
    
    for i in range(POINTCLOUD_AUGMENT_NUM):
        if i % 2 == 0:
            augment_mat = augment_data()
        else:
            augment_mat = augment_data(flip=True)
        augment_mats.append(augment_mat)

    ggarray, cloud, points_down, grasp_features, sinput = get_grasp(net, depths, existing_shm_color, augment_mat=augment_mat1)
    for i in range(POINTCLOUD_AUGMENT_NUM):
        if i % 2 == 0:
            ggarray2, _, _, grasp_features2, sinput2 = get_grasp(net, depths, existing_shm_color, augment_mat=augment_mats[i])
        else:
            ggarray2, _, _, grasp_features2, sinput2 = get_grasp(net, depths, existing_shm_color, augment_mat=augment_mats[i], flip=True)
        if ggarray2 is None:
            continue
        if ggarray is None:
            ggarray = ggarray2
            grasp_features = grasp_features2
            sinput = sinput2
        else:
            sinput.append(sinput2[0])
            ggarray = torch.cat([ggarray, ggarray2], axis=0)
            grasp_features = torch.cat([grasp_features, grasp_features2], axis=0)
    return ggarray, cloud, points_down, grasp_features, sinput

def select_grasp_type(inspire_gg):
    select_gg_types = [[] for _ in range(1, NUM_OF_INSPIRE_TYPE+1)]
    for idx, score in enumerate(inspire_gg.scores):
        grasp_type = inspire_gg.grasp_types[idx]
        if grasp_type in [1]:
            max_num = 10
        else:
            max_num = 50
        if len(select_gg_types[int(grasp_type)]) < max_num:
            select_gg_types[int(grasp_type)].append(idx)
    gg_type_id = []
    for gg_type in select_gg_types:
        gg_type_id += gg_type
    return gg_type_id

def robot_grasp(cfgs):
    net = get_net(cfgs.checkpoint_path, use_v2=cfgs.use_graspnet_v2)
    robot = get_robot(cfgs.robot_ip, robot_debug=True, gripper_type='InspireHandR', global_cam=cfgs.global_camera)
    inspire_models = get_inspire_model(cfgs.inspire_model_path)
    fail = 0
    existing_shm_color = shared_memory.SharedMemory(name='realsense_color')
    existing_shm_depth = shared_memory.SharedMemory(name='realsense_depth')
    meshes_pcls = load_meshes_pointcloud(cfgs.inspire_mesh_json_path)

    try:
        v = 0.07
        a = 0.07
        if cfgs.global_camera:
            robot.movej(robot.throwj2, acc=a*2,
                        vel=v*3)  # this v and a are anguler, so it should be larger than translational
            t1 = time.time()
            depths = get_depth(existing_shm_depth)
            depths_saved = copy.deepcopy(depths)
            colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
            ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(existing_shm_depth, existing_shm_color, net)
            t3 = time.time()
            print(f'Net Time:{t3 - t1}')

        else:
            robot.movel(robot.ready_pose(), acc=a*2,
                        vel=v*3)  # this v and a are anguler, so it should be larger than translational

        while True:

            if not cfgs.global_camera:
                t1 = time.time()
                robot.movel(robot.ready_pose(), acc=a * 2, vel=v * 3,
                            wait=True)  # this v and a are anguler, so it should be larger than translational
                time.sleep(0.5)
                print('movel')
                time.sleep(0.7)
                depths = get_depth(existing_shm_depth)
                depths_saved = copy.deepcopy(depths)
                colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
                t1 = time.time()
                ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(existing_shm_depth,
                                                                                   existing_shm_color, net)
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
                    depths_saved = copy.deepcopy(depths)
                    colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
                    ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(existing_shm_depth,
                                                                                       existing_shm_color, net)
                    time.sleep(0.1)
                if DEBUG:
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                    o3d.visualization.draw_geometries([cloud, frame, sphere])
                continue

            ########## PROCESS GRASPS ##########
            # collision detection
            ggarray = ggarray.cpu().numpy()
            
            # Prevent the robot arm from crossing the border, 
            grasp_features = grasp_features.cpu().numpy()
            two_fingers_source_grasp_features = copy.deepcopy(grasp_features)

            ggarray, if_flip = flip_ggarray(ggarray)
            grasp_features = np.c_[grasp_features, if_flip]
            ggarray = ggarray[~np.array(if_flip)]
            grasp_features = grasp_features[~np.array(if_flip)]

            source_index = ggarray[:, 0].argsort()
            ggarray = ggarray[source_index][::-1][:1000]
            grasp_features = grasp_features[source_index][::-1][:1000]
            t_multi = time.time()
            grasp_features_dic = get_graspgroup_features(grasp_features, sinput)
            
            inspire_depth, inspire_type, scores, ggarray, grasp_features = \
                                            get_inspire_depth_type(inspire_models, grasp_features_dic,
                                                                    ggarray, grasp_features=grasp_features)

            if not RANDOM_GRASP:
                score_thresh = 0.85
                mask = (scores > score_thresh)
                ggarray = ggarray[mask]
                grasp_features = grasp_features[mask]
                inspire_depth = inspire_depth[mask]
                inspire_type = inspire_type[mask]
                scores = scores[mask]
            
            if len(ggarray) == 0:
                print('There is no grasp that score greater than 0.9 ')
                if cfgs.global_camera:
                    depths = get_depth(existing_shm_depth)
                    depths_saved = copy.deepcopy(depths)
                    colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
                    ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(existing_shm_depth, existing_shm_color, net)
                if DEBUG:
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                    o3d.visualization.draw_geometries([cloud, frame, sphere])
                continue

            two_fingers_ggarray = GraspGroup(ggarray)
            two_fingers_ggarray_object_ids_source = ggarray[:, 16]
            two_fingers_ggarray_source = copy.deepcopy(two_fingers_ggarray)

            InspireHandR_ggarray = InspireHandRGraspGroup() 
            InspireHandR_ggarray.set_grasp_min_width(MIN_GRASP_WIDTH)
            InspireHandR_ggarray.from_graspgroup(two_fingers_ggarray, inspire_type, cfgs.inspire_mesh_json_path)
            InspireHandR_ggarray.object_ids = two_fingers_ggarray_object_ids_source
            InspireHandR_ggarray.scores = scores
            InspireHandR_ggarray.depths = InspireHandR_ggarray.depths + inspire_depth + INSPIREHANDR_DEFAULT_DEPTH
            InspireHandR_ggarray_source = copy.deepcopy(InspireHandR_ggarray)

            index_filter_by_z_axis = InspireHandR_ggarray.filter_grasp_group_by_z_axis(0.4)
            two_fingers_ggarray = two_fingers_ggarray[index_filter_by_z_axis]
            two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids_source[index_filter_by_z_axis]
            grasp_features = grasp_features[index_filter_by_z_axis]

            if len(InspireHandR_ggarray) == 0:
                print('No grasp detected after filter')
                if cfgs.global_camera:
                    ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(existing_shm_depth, existing_shm_color, net)
                if DEBUG:
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                    o3d.visualization.draw_geometries([cloud, frame, sphere])
                continue
            
            index_score = np.argsort(InspireHandR_ggarray.scores)[::-1]
            InspireHandR_ggarray = InspireHandR_ggarray[index_score]
            two_fingers_ggarray = two_fingers_ggarray[index_score]
            grasp_features = grasp_features[index_score]
            two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_score]

            index_type = select_grasp_type(InspireHandR_ggarray)
            InspireHandR_ggarray = InspireHandR_ggarray[index_type]
            two_fingers_ggarray = two_fingers_ggarray[index_type]
            grasp_features = grasp_features[index_type]
            two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_type]

            approach_distance = 0.06
            start_time = time.time()
            mfcdetector = ModelFreeCollisionDetectorMultifinger(points_down.cpu().numpy(), voxel_size=0.001)
            InspireHandR_ggarray, two_fingers_ggarray, empty_mask, min_width_index = mfcdetector.detect(InspireHandR_ggarray, two_fingers_ggarray,
                                                                  cfgs.inspire_mesh_json_path, meshes_pcls, min_grasp_width=MIN_GRASP_WIDTH,
                                                                  VoxelGrid=INSPIREHANDR_VOXElGRID, DEBUG=False, approach_dist=approach_distance,
                                                                  collision_thresh=0, adjust_gripper_centers=True,)

            # proposals
            InspireHandR_ggarray = InspireHandR_ggarray[empty_mask]
            two_fingers_ggarray = two_fingers_ggarray[empty_mask]
            two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[min_width_index][empty_mask]
            grasp_features = grasp_features[min_width_index][empty_mask]

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
                    ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(existing_shm_depth,
                                                                                       existing_shm_color, net)
                    t3 = time.time()
                    print(f'Net Time:{t3 - t1}')
                continue

            # sort
            index_score = np.argsort(InspireHandR_ggarray.scores)[::-1][:10]
            two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_score]
            InspireHandR_ggarray = InspireHandR_ggarray[index_score]
            two_fingers_ggarray = two_fingers_ggarray[index_score]
            grasp_features = grasp_features[index_score]

            idx = random.randint(0, len(index_score)-1)
            InspireHandR_grasp_used = InspireHandR_ggarray[idx]
            two_fingers_grasp_used = two_fingers_ggarray[idx]
            grasp_features_used = grasp_features[idx]


            print('picked by scores rotations, translations: ', InspireHandR_grasp_used.rotation_matrix,
                  InspireHandR_grasp_used.translation, two_fingers_grasp_used.translation, two_fingers_grasp_used.rotation_matrix)
            print('grasp score:', InspireHandR_grasp_used.score, two_fingers_grasp_used.score)
            print('grasp width:', InspireHandR_grasp_used.width, two_fingers_grasp_used.width)
            print('grasp depth:', InspireHandR_grasp_used.depth, two_fingers_grasp_used.depth)
            print('grasp type:', InspireHandR_grasp_used.grasp_type)
            print('grasp angle:', InspireHandR_grasp_used.angle)

            t4 = time.time()
            print(f'Collision Processing Time:{t4 - t3}')
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

            gripper_time = 0.4
            robot.open_gripper(InspireHandR_grasp_used.angle)
            mat_pose = robot.grasp_and_throw(InspireHandR_grasp_used, two_fingers_grasp_used, cloud, cfgs.inspire_mesh_json_path,
                                             acc=a*2, vel=v*3, approach_dist=approach_distance,
                                             execute_grasp=True, use_ready_pose=True, gripper_time=gripper_time)

            while robot.is_program_running():
                pass

            t45 = time.time()
            if cfgs.global_camera:
                robot.movej(robot.throwj2, acc=a * 4, vel=v * 5.5)  # this v and a are anguler, so it should be larger than translational
                robot.open_gripper(InspireHandR_grasp_used.angle)

            save_grasp_information(two_fingers_ggarray, two_fingers_ggarray_object_ids, InspireHandR_ggarray, 
                            two_fingers_ggarray_source, InspireHandR_ggarray_source, two_fingers_ggarray_object_ids_source,
                            two_fingers_grasp_used, InspireHandR_grasp_used, grasp_features_used, 
                            mat_pose, colors_saved, depths_saved, grasp_features)

            if cfgs.global_camera:
                t1 = time.time()
                depths = get_depth(existing_shm_depth)
                depths_saved = copy.deepcopy(depths)
                colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
                ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(existing_shm_depth,
                                                                                   existing_shm_color, net)
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
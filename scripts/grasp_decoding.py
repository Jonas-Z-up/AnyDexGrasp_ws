"""Checkpoint-compatible decoding, extracted from scripts/robot_inspire.py.
No robot/camera initialization. Network tensor layout and 480-D features preserved.
Local fixes: actually apply seed objectness, and use an explicit cross-product dimension.
Source SHA256: 3cd5e7e55d1c536d8e9f191ff9abed00a69ef7f5c93eb064d1adca11aa62e70e
"""
import copy
from collections import OrderedDict
import numpy as np
import torch
def batch_viewpoint_params_to_matrix(towards, angle):
    """Explicit vector dimension also works when exactly three seeds remain."""
    x = towards / torch.norm(towards, dim=-1, keepdim=True)
    zero = torch.zeros_like(x[:, 0])
    y = torch.stack((-x[:, 1], x[:, 0], zero), dim=-1)
    y[torch.norm(y, dim=-1) == 0, 1] = 1
    y = y / torch.norm(y, dim=-1, keepdim=True)
    z = torch.cross(x, y, dim=-1)
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((x, y*c[:, None]+z*s[:, None], -y*s[:, None]+z*c[:, None]), dim=-1)
BATCH_SIZE = 1
MAX_GRASP_WIDTH = .1
NUM_OF_INSPIRE_TYPE = 8
NUM_OF_INSPIRE_DEPTH = 4
RANDOM_GRASP = False
def parse_preds(end_points, use_v2=False, filter_objectness=True):
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
    for i in range(seed_xyz.shape[0]):
        
        cloud_mask_i = (coords[:, 0] == i)
        seed_inds_i = seed_inds[i]
        objectness_mask_i = objectness_mask[cloud_mask_i][seed_inds_i]  # (Ns,)

        if filter_objectness and not objectness_mask_i.any():
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

        # The legacy script computed objectness but left all per-seed indexing
        # commented out. Keep tensors aligned while excluding background seeds.
        if filter_objectness:
            seed_xyz_i = seed_xyz_i[objectness_mask_i]
            point_features_i = point_features_i[objectness_mask_i]
            seed_inds_i = seed_inds_i[objectness_mask_i]
            before_generator_i = before_generator_i[objectness_mask_i]
            grasp_view_xyz_i = grasp_view_xyz_i[objectness_mask_i]
            grasp_view_inds_i = grasp_view_inds_i[objectness_mask_i]
            grasp_view_scores_i = grasp_view_scores_i[objectness_mask_i]
            grasp_scores_i = grasp_scores_i[objectness_mask_i]
            grasp_widths_i = grasp_widths_i[objectness_mask_i]
        
        Ns, A, D = grasp_scores_i.size()
        grasp_features_two_finger_i = grasp_features_two_finger[i] # [objectness_mask_i] # (Ns', 3 + C)
        if filter_objectness:
            grasp_features_two_finger_i = grasp_features_two_finger_i[objectness_mask_i]
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

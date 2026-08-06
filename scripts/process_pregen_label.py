import numpy as np
import pickle
from tqdm import tqdm
import os

camera = 'realsense'
root = '/aidata/graspnet_v1_newformat/'  # TODO: Change this to your GraspNet dataset path if training
sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in range(71,130)]

for i,x in enumerate(tqdm(sceneIds, desc = 'Loading data path and collision labels...')):
    graspness = []
    visibleid = []
    for img_num in range(256):
        view_graspness = {}
        graspness_tmp = {}

        if not os.path.exists(os.path.join('/aidata', 'pre_generated_label_haoshu_antipodalscore', x, camera)):
            os.makedirs(os.path.join('/aidata', 'pre_generated_label_haoshu_antipodalscore', x, camera))
        graspness_files = os.listdir(os.path.join(root, 'pre_generated_label_antipodalscore', x, camera, str(img_num).zfill(4), 'graspness'))
        for graspness_file in graspness_files:
            tmp_file = np.load(os.path.join(root, 'pre_generated_label_antipodalscore', x, camera, str(img_num).zfill(4), 'graspness', graspness_file))
            graspness_tmp[str(int(graspness_file[:3])+1)+'_pointwise_graspness'] = tmp_file['pointwise_graspness'] #obj id + 1 to align with seg image label
            view_graspness[str(int(graspness_file[:3])+1)+'_viewwise_graspness'] = tmp_file['viewwise_graspness']
        graspness.append(graspness_tmp)
        np.save(os.path.join('/aidata', 'pre_generated_label_haoshu_antipodalscore', x, camera, str(img_num).zfill(4)+'_view_graspness.npy'), view_graspness)
        

        visibleid_tmp = {}
        visibleid_files = os.listdir(os.path.join(root, 'pre_generated_label_antipodalscore', x, camera, str(img_num).zfill(4), 'visible_indices'))
        for visibleid_file in visibleid_files:
            visibleid_tmp[str(int(visibleid_file[:3])+1)] = np.load(os.path.join(root, 'pre_generated_label_antipodalscore', x, camera, str(img_num).zfill(4), 'visible_indices', visibleid_file)) #obj id + 1 to align with seg image label
        visibleid.append(visibleid_tmp)

    np.save(os.path.join('/aidata', 'pre_generated_label_haoshu_antipodalscore', x, camera, 'point_graspness.npy'), graspness)
    np.save(os.path.join('/aidata', 'pre_generated_label_haoshu_antipodalscore', x, camera, 'visibleid.npy'), visibleid)

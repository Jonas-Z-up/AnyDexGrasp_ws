import copy
import json
import os
import sys
import time

def update_grasp_type(json_path, gripper):
    dh3_mapping = {0:1,1:2,2:3,3:4}
    inspire_mapping = {0:1,1:2,2:3,3:4,4:5,5:-1,6:6,7:7,8:-1,9:8,10:-1,11:-1}
    allegro_mapping = {0:1,1:2,2:3,3:4,4:5,5:6,6:-1,7:-1,8:7,9:8,10:9,11:-1,12:10}
    if gripper == 'dh3':
        gripper_mapping = dh3_mapping
        from ur_toolbox.robot.DH3.DH3_grasp import grasp_types
        name = 'DH3'
    elif gripper == 'allegro':
        gripper_mapping = allegro_mapping
        from ur_toolbox.robot.Allegro.Allegro_grasp import grasp_types
        name = 'Allegro'
    elif gripper == 'inspire':
        gripper_mapping = inspire_mapping   
        from ur_toolbox.robot.Inspire.InspireHandR_grasp import grasp_types 
        name = 'InspiredHandR'
    with open(json_path, 'r') as f:
        information = json.load(f)
    remove_item = []
    for date, date_info in information.items():
        for k,v in date_info.items():
            if k == name + '_pose_finger_type':
                information[date][k] = int(gripper_mapping[information[date][k]])
                if information[date][k] == -1:
                    remove_item.append(date)
            if k == 'collision_grasp_feature':
                remove_item_c = []
                for idx, collision_item in enumerate(v):
                    for k_c, v_c in collision_item.items():
                        if k_c == name + '_pose_finger_type':
                            information[date][k][idx][k_c] = int(gripper_mapping[information[date][k][idx][k_c]])
                            if information[date][k][idx][k_c] == -1:
                                remove_item_c.append(collision_item)
                        for k_c, v_c in collision_item.items():
                            if k == '_pose':
                                information[date][k][idx][k_c][2] = information[date][k][idx][name + '_pose_finger_type']
                for x in remove_item_c:
                    information[date][k].remove(x)
        for k,v in date_info.items():            
            if k == name + '_pose':
                information[date][k][2] = information[date][name + '_pose_finger_type']
    for x in remove_item:
        information.pop(x)   
    json_file = json.dumps(information, indent=4)
    with open(json_path, 'w') as handle:
        handle.write(json_file)

            
def read_files(path, gripper):
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            json_path = os.path.join(fpathe,f)
            print(json_path)
            update_grasp_type(json_path, gripper)

def merge_data(in_path, out_path, gripper='Allegro'):
    informations = dict()
    grasp_types = os.listdir(in_path)
    for grasp_type in grasp_types:
        grasp_type_path = os.path.join(in_path, grasp_type)
        for date in os.listdir(grasp_type_path):
            date_path = os.path.join(grasp_type_path, date, 'information.json')
            with open(date_path, 'r') as f:
                information = json.load(f)
            informations[date] = information
            del information['two_fingers_pose_AD']
            del information['two_fingers_pose_features']
            del information['two_fingers_pose_features_before_generator']
            del information['point_features']
            del information['before_collision']
            del information['after_collision']
            del information['base_2_tcp1']
            del information['base_2_tcp1_backup']
            del information['tcp_2_gripper']
            del information['base_2_TwoFingersGripper_pose']
            del information['tcp_2_camera']
            del information['base_2_tcp_ready']
            del information['camera_internal']
            del information['two_fingers_ggarray_proposals']
            del information[gripper+'_ggarray_proposals']
            del information['two_fingers_ggarray_informations_proposals']
            del information['two_fingers_ggarray_source']
            del information[gripper+'_ggarray_source_saved']
    json_file = json.dumps(informations, indent=4)
    with open(out_path, 'w') as handle:
        handle.write(json_file)


if __name__ == '__main__':
    # inspire_path = 'logs/data/decision_model/inspire'
    # read_files(inspire_path, 'inspire')
    # dh3_path = 'logs/data/decision_model/dh3'
    # read_files(dh3_path, 'dh3')
    # allegro_path = 'logs/data/decision_model/allegro'
    # read_files(allegro_path, 'allegro')
    in_path = 'logs/data/decision_model/allegro/obj140/collect'
    out_path = 'logs/data/decision_model/allegro/obj140.json'
    merge_data(in_path, out_path, 'Allegro')
    
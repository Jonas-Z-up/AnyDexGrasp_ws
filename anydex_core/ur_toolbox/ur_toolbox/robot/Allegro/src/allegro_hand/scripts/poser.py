#!/usr/bin/python3
# Basic imports
import rospy
import numpy as np
import yaml

# Allegro hand controller imports
from allegro_hand.controller import AllegroController
from allegro_hand.utils import find_file

def perform_poses(yaml_file):
    # Initializing the controller
    allegro_controller = AllegroController()

    # Loading the actions from YAML file
    actions = []
    with open(yaml_file, 'r') as file:
        yaml_file = yaml.full_load(file)
        for key, array in yaml_file.items():
            actions.append(array)

    actions = np.array(actions)
    allegro_ready_pose = np.array([[0, 0.6, 0.6, 0.6], [0, 0.6, 0.6, 0.6], [0, 0.6, 0.6, 0.6], [1.496, 0, 0.35, 0.35]]).reshape(16)
    allegro_ready_pose1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1.496, 0, 0, 0]]).reshape(16)
    torque = np.array([[0, 0, 0., 0.], [0, 0.3, 0.25, 0.2], [0, 0.3, 0.25, 0.2], [0, 0., 0.25, 0.2]]).reshape(16)
    # # while True:
    # # allegro_controller.apply_joint_torque(np.zeros(16))
    allegro_controller.hand_pose(allegro_ready_pose, True)
    allegro_controller.apply_joint_torque(torque*0.33)
    rospy.sleep(8)
    allegro_controller.hand_pose(allegro_ready_pose1, True)
    allegro_controller.apply_joint_torque(torque)
    # rospy.sleep(5)
    # print(allegro_controller.current_joint_pose.position)
        # allegro_controller.hand_pose(allegro_ready_pose1, True)
        # rospy.sleep(5)
        # print(allegro_controller.current_joint_pose.effort)
    # Performing all the actions
    # for iterator in range(len(actions)):
    #     print('Hand is performing pose:', iterator + 1)
    #     allegro_controller.hand_pose(actions[iterator], True)
        
    #     print('Pausing so that the robot reaches the commanded hand pose...\n')
    #     rospy.sleep(2)

    print('Finished all the poses!')

if __name__ == '__main__':
    # Finding the parameter package path
    yaml_file_path = find_file("allegro_hand_parameters", "poses.yaml")

    # Performing the poses
    perform_poses(yaml_file_path)
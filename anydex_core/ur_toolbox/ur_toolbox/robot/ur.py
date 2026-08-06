import os
import copy
import math
import json
import time
from math import pi

import cv2
import numpy as np
import urx
import xlrd2

from ..camera import RealSense
from ..transformation.pose import pose_array_2_matrix, pose_matrix_2_array, translation_rotation_2_matrix, \
    translation_rotation_2_array

from .robotiq import Robotiq
from .wsg import WSG
from .Inspire.InspireHandR import InspireHandR
try:
    from .DH3.DH3 import DH3
except ImportError:
    DH3 = None
try:
    from .Allegro.Allegro import Allegro
except ImportError:
    Allegro = None
# import ikfastpy

from graspnetAPI import Grasp
from .Inspire.InspireHandR_grasp import InspireHandRGrasp
try:
    from .DH3.DH3_grasp import DH3Grasp
except ImportError:
    DH3Grasp = None
try:
    from .Allegro.Allegro_grasp import AllegroGrasp, grasp_types
except ImportError:
    AllegroGrasp = None

def to_str(var):
    return '[' + str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1] + ']'


class UR_Camera_Gripper(urx.Robot):
    DEFAULT_CAM2BASE_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "calibration", "cam2base_current.json")
    )

    def __init__(self, host, use_rt=False, camera=None, robot_debug=True, gripper_on_robot=False,
                 gripper_type='robotiq', gripper_port='/dev/ttyUSB0', global_cam=False):
        '''
        **Input:**
        - host: string of the ip address of the robot
        - use_rt: use real time mode or not. get_force is only available in realtime mode. Consumes CPU.
        - camera: if "realsense" use old camera interface which take 5 pictures and use the last one.
        - robot_debug: if False, the camera frame is in the depth image. if True,  in the rgb image.
        - RT: the rotation and translation
        '''
        self.robot_debug = robot_debug
        self.host = host
        self.gripper_port = gripper_port
        self.global_cam = global_cam
        self._cam2base_matrix = None
        super(UR_Camera_Gripper, self).__init__(host, use_rt)
        self.gripper_on_robot = gripper_on_robot
        self.gripper_type = gripper_type
        if self.gripper_on_robot:
            self.gripper = Robotiq_Two_Finger_Gripper(self)
            self.gripper_action = self.gripper.gripper_action
            self.open_gripper = self.gripper.open_gripper
            self.close_gripper = self.gripper.close_gripper
            self.gripper_home = self.gripper.open_gripper
        elif self.gripper_type == 'robotiq':
            self.gripper = Robotiq(port=self.gripper_port)
            self.get_gripper_position = self.gripper.get_gripper_position
            self.gripper_action = self.gripper.gripper_action
            self.open_gripper = self.gripper.open_gripper
            self.close_gripper = self.gripper.close_gripper
            self.gripper_home = self.gripper.open_gripper
        elif self.gripper_type == 'wsg':
            self.gripper = WSG(TCP_IP=self.gripper_port)
            self.get_gripper_position = self.gripper.get_gripper_position
            self.gripper_action = self.gripper.gripper_action
            self.open_gripper = self.gripper.open_gripper
            self.close_gripper = self.gripper.close_gripper
            self.gripper_home = self.gripper.home
        elif self.gripper_type == 'InspireHandR':
            self.gripper = InspireHandR(port=self.gripper_port)
            self.set_clear_error = self.gripper.set_clear_error
            self.open_gripper = self.gripper.open_gripper
            self.close_gripper = self.gripper.close_gripper
            self.gripper_home = self.gripper.reset
        elif self.gripper_type == 'DH3':
            self.gripper = DH3(ip=self.gripper_port)
            self.open_gripper = self.gripper.open_gripper
            self.close_gripper = self.gripper.close_gripper
            self.gripper_home = self.gripper.set_ready_pose
        elif self.gripper_type == 'Allegro':
            self.gripper = Allegro()
            self.open_gripper = self.gripper.open_gripper
            self.close_gripper = self.gripper.close_gripper
            self.gripper_home = self.gripper.set_ready_pose
            self.set_torque = self.gripper.set_torque
        else:
            raise NotImplementedError
        self.camera = camera
        if self.camera is 'realsense':
            self.camera = RealSense()
            self.get_rgbd_image = self.camera.get_rgbd_image
            self.get_rgb_image = self.camera.get_rgb_image
            self.get_depth_image = self.camera.get_depth_image

        self.readyj = np.array(
            [-0.20474416414369756, -1.380301300679342, 0.9994006156921387, -1.1843579451190394, -1.5655363241778772,
             -0.21112090746034795])

        self.waypointj = np.array(
            [-0.2048280874835413, -1.4410565535174769, 1.33439302444458, -1.4585440794574183, -1.5655601660357874,
             -0.21124059358705694])

        self.throwj = np.array(
            [0.726066529750824, -1.1951254049884241, 1.1765141487121582, -1.6369522253619593, -1.49585467973818,
             -0.044919792805806935])
        self.throwj2 = np.array([0.6649638414382935, -1.4355509916888636, 1.5572257041931152, -2.7616093794452112, 
                                -1.4952309767352503, -1.3120296637164515])

    def get_camera_tcp_matrix(self):
        '''
        **Output:**
        - numpy array of shape (4,4) of the camera tcp transformation matrix
        '''
        # Relative offset from camera center to the gripper center.
        # The realsense center is on one of the two cameras.
        # The center of robotiq gripper is 4cm above the lowest point of gripper.

        # x to left from base is minus, y to forward from base is minus, z to higher is minus
        if not self.robot_debug:
            return np.array([[1, 0, 0, -0.009], [0, 1, 0, -0.05], [0, 0, 1, -0.13], [0, 0, 0, 1]], dtype=np.float32)
        elif self.gripper_type == 'robotiq':
            return np.array([[1, 0, 0, -0.027], [0, 1, 0, -0.075], [0, 0, 1, -0.170], [0, 0, 0, 1]], dtype=np.float32)
        elif self.gripper_type == 'wsg':
            return np.array([[1, 0, 0, -0.020], [0, 1, 0, -0.083], [0, 0, 1, -0.153], [0, 0, 0, 1]], dtype=np.float32)
        elif self.gripper_type == 'InspireHandR':
            return np.array([[1, 0, 0, -0.025], [0, 1, 0, -0.07], [0, 0, 1, -0.0406], [0, 0, 0, 1]], dtype=np.float32)
        elif self.gripper_type == 'DH3':
            return np.array([[1, 0, 0, -0.027], [0, 1, 0, -0.114], [0, 0, 1, 0.0608], [0, 0, 0, 1]], dtype=np.float32)
        elif self.gripper_type == 'Allegro':
            return np.array([[1, 0, 0, -0.042], [0, 1, 0, -0.11], [0, 0, 1, 0.016], [0, 0, 0, 1]], dtype=np.float32)
        else:
            raise NotImplementedError

    def load_cam2base_matrix(self, calibration_path=None):
        '''
        **Input:**
        - calibration_path: optional path to JSON/NPZ camera-to-base calibration.
        **Output:**
        - numpy array of shape (4,4) of the camera to base transformation matrix.
        '''
        calibration_path = calibration_path or self.DEFAULT_CAM2BASE_PATH
        if calibration_path.endswith('.npz'):
            data = np.load(calibration_path)
            rotation = np.asarray(data['R_cam2base'], dtype=np.float32)
            translation = np.asarray(data['t_cam2base'], dtype=np.float32)
        else:
            with open(calibration_path, 'r') as f:
                data = json.load(f)
            rotation = np.asarray(data['R_cam2base'], dtype=np.float32)
            translation = np.asarray(data['t_cam2base'], dtype=np.float32)
        return translation_rotation_2_matrix(translation, rotation).astype(np.float32)

    def get_cam2base_matrix(self):
        '''
        **Output:**
        - numpy array of shape (4,4) of the camera to base transformation matrix.
        '''
        if self._cam2base_matrix is None:
            self._cam2base_matrix = self.load_cam2base_matrix()
        return self._cam2base_matrix.copy()

    def get_gripper_tcp_matrix(self):
        '''
        **Output:**
        - numpy array of shape (4,4) of the gripper tcp transformation matrix
        '''
        if self.gripper_type == 'InspireHandR':
            return np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        elif self.gripper_type == 'DH3':
            return np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0.058], [0, 0, 0, 1]], dtype=np.float32)
        elif self.gripper_type == 'Allegro':
            return np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0.049], [0, 0, 0, 1]], dtype=np.float32)

    def get_target_gripper_base_pose(self, gripper_camera_pose, use_ready_pose=False):
        '''
        **Input:**
        - gripper_camera_pose: np.array of shape (4,4) of the gripper pose in camera coordinate.
        **Output:**
        - target_gripper_base_pose: target gripper pose in base coordinate, shape=(4,4)
        '''
        if self.global_cam:
            return np.dot(self.get_cam2base_matrix(), gripper_camera_pose)
        return np.dot(
            np.dot(
                self.get_tcp_base_matrix(use_ready_pose=use_ready_pose),
                self.get_camera_tcp_matrix()
            ),
            gripper_camera_pose,
        )

    def gripper_camera_pose_2_tcp_base_pose(self, gripper_camera_pose, use_ready_pose=False):
        '''
        **Input:**
        - gripper_camera_pose: np.array of shape (4,4) of the gripper pose in camera coordinate.
        **Output:**
        - tcp_base_pose: target tcp pose in base coordinate, shape=(4,4)
        '''
        if self.global_cam:
            tcp_pose = np.dot(
                np.dot(
                    self.get_cam2base_matrix(),  # base / camera
                    gripper_camera_pose  # camera / gripper
                ),
                np.linalg.inv(self.get_gripper_tcp_matrix())  # (tcp2 / gripper)^(-1)
            )
            return tcp_pose

        tcp_pose = np.dot(
            np.dot(
                np.dot(
                    self.get_tcp_base_matrix(use_ready_pose=use_ready_pose),  # base / tcp1
                    self.get_camera_tcp_matrix()  # tcp1 / camera
                ),
                gripper_camera_pose  # camera / gripper
            ),
            np.linalg.inv(self.get_gripper_tcp_matrix())  # (tcp2 / gripper)^(-1)
        )

        return tcp_pose

    def tcp_base_pose_2_gripper_camera_pose(self, tcp_base_pose):
        '''
        **Input:**
        - tcp_base_pose: np.array of shape (4,4) of the tcp pose in base coordinate.
        **Output:**
        - gripper_camera_pose: target gripper pose in camera coordinate, shape=(4,4)
        '''
        if self.global_cam:
            return np.dot(
                np.dot(
                    np.linalg.inv(self.get_cam2base_matrix()),  # base / camera
                    tcp_base_pose  # base / tcp2
                ),
                self.get_gripper_tcp_matrix(),  # (tcp2 / gripper)^(-1)
            )  # camera / gripper
        return np.dot(
            np.dot(
                np.dot(
                    np.linalg.inv(self.get_camera_tcp_matrix()),  # tcp 1/ camera
                    np.linalg.inv(self.get_tcp_base_matrix())  # base / tcp1
                ),
                tcp_base_pose  # base / tcp2
            ),
            self.get_gripper_tcp_matrix(),  # (tcp2 / gripper)^(-1)
        )  # camera / gripper

    def ready_pose(self):
        '''
        **Output:**
        - np.array of shape (6) of the robot pose in ready state.
        '''
        if self.global_cam:
            return np.array([-0.57084448065005, 0.0067573021191121875, 0.620, 2.290100059688817, 2.141294337325684, -0.000345113451578144], dtype = np.float32)
        else:
            return np.array([-0.299333228670771, 0.0022835537920474144, 0.603633972814212, 2.2151031430447197, 2.2263986774485334, 0.00013764345054940088], dtype=np.float32)

    def search_pose(self, num=0):
        '''
        **Output:**
        - np.array of shape (6) of the robot pose in ready state.
        '''
        if num == 0:
            return np.array(
                [-0.5469021157445321, 0.1347367254046668, 0.35365830507719254, 2.2080679519699693, 2.2255254359753684,
                 -0.00904318571969405], dtype=np.float32)
        elif num == 1:
            return np.array(
                [-0.5437058596698867, 0.04709940798610358, 0.45000698738643363, 2.207945931720303, 2.2256528832802553,
                 -0.00895043340604062], dtype=np.float32)
        elif num == 2:
            return np.array(
                [-0.5437224468013898, 0.047160105398491296, 0.29994811927492937, 2.207828720048168, 2.225748636740982,
                 -0.008737462058821683], dtype=np.float32)
        elif num == 3:
            return np.array(
                [-0.5469137229836639, -0.18049111345598381, 0.35366300049701666, 2.208033888119672, 2.2255688673472203,
                 -0.009075800937384288], dtype=np.float32)
        else:
            return np.array(
                [-0.5437105916764003, -0.14394231592830342, 0.4500524872992883, 2.2080937342456632, 2.2257196198809557,
                 -0.009168278193525125], dtype=np.float32)

    def throw_pose(self):
        '''
        **Output:**
        - np.array of shape (6) of the robot pose in ready state.
        '''
        if self.global_cam:
            return np.array(
                [-0.410796973800367, -0.5401806313280444, 0.22974717091039015, 1.202156341186167, 2.8652524529413563,
                 -0.16828893640775433], dtype=np.float32)
        else:
            return np.array([-0.3204891342444824, 0.0, 0.60397890768936943, 2.2151031430447197, 2.2263986774485334, 0.013764345054940088], dtype = np.float32)

    def throw(self, acc=0.05, vel=0.05):
        '''
        **Output:**
        - no output but put the robot to throw pose and open the gripper.
        '''
        self.movel(self.throw_pose(), acc=acc, vel=vel)

    def ready(self, acc=0.05, vel=0.05):
        '''
        **Output:**
        - no output but put the robot to ready state and open the gripper.
        '''
        self.movel(self.ready_pose(), acc=acc, vel=vel)
        self.gripper_home()

    def search(self, acc=0.05, vel=0.05, n=0):
        '''
        **Output:**
        - no output but put the robot to ready state and open the gripper.
        '''
        self.movel(self.search_pose(n), acc=acc, vel=vel)
        self.gripper_home()

    def normalize(self, x):
        return np.array([x[0], x[1], x[2]]) / math.sqrt(np.power(x[0], 2) + np.power(x[1], 2) + np.power(x[2], 2))

    def execute(self, grasp, acc=0.05, vel=0.05, approach_dist=0.10):
        '''
        **Input:**
        - grasp: Grasp instance or numpy array of shape (4,4) or numpy array of shape (6,) in base coordinate.
        - acc: float of the maximum acceleration.
        - vel: float of the maximum velocity.
        - approach_dist: float of the distance to move along the z axis of tcp coordinate.
        **Output:**
        - No output but the robot moves to the given pose.
        '''
        if isinstance(grasp, Grasp):
            translation = grasp.translation
            rotation = grasp.rotation_matrix
            pose = translation_rotation_2_array(translation, rotation)
        elif isinstance(grasp, np.ndarray):
            if grasp.shape == (4, 4):
                pose = pose_matrix_2_array(grasp)
            elif grasp.shape == (6,):
                pose = grasp
            else:
                raise ValueError('Shape of Grasp Array must be (4,4) or (6,), but it is {}'.format(grasp.shape))
        else:
            raise ValueError('execute must be called with Grasp or numpy array, but it is {}'.format(type(grasp)))
        tcp_pose = pose_array_2_matrix(pose)
        tcp_pre_pose = copy.deepcopy(tcp_pose)
        tcp_pre_pose[:3, 3] = tcp_pre_pose[:3, 3] - approach_dist * tcp_pre_pose[:3, 2]
        self.movel(tcp_pre_pose, acc=acc, vel=vel)
        self.movel(tcp_pose, acc=acc, vel=vel)

    def execute_camera_pose(self, grasp, acc=0.05, vel=0.05, approach_dist=0.10):
        '''
        **Input:**
        - grasp: Grasp instance or numpy array of shape (4,4) or numpy array of shape (6,) in camera coordinate
        - acc: float of the maximum acceleration.
        - vel: float of the maximum velocity.
        - approach_dist: float of the distance to move along the z axis of tcp coordinate.
        **Output:**
        - No output but the robot moves to the ready pose first, and then moves to the given pose along the z axis of the tcp coordinate.
        '''
        if isinstance(grasp, Grasp):
            translation = grasp.translation
            rotation = grasp.rotation_matrix
            pose = translation_rotation_2_array(translation, rotation)
        elif isinstance(grasp, np.ndarray):
            if grasp.shape == (4, 4):
                pose = pose_matrix_2_array(grasp)
            elif grasp.shape == (6,):
                pose = grasp
            else:
                raise ValueError('Shape of Grasp Array must be (4,4) or (6,), but it is {}'.format(grasp.shape))
        else:
            raise ValueError('execute must be called with Grasp or numpy array, but it is {}'.format(type(grasp)))
        pose = pose_array_2_matrix(pose)
        tcp_pose = self.gripper_camera_pose_2_tcp_base_pose(pose)
        tcp_pre_pose = copy.deepcopy(tcp_pose)
        tcp_pre_pose[:3, 3] = tcp_pre_pose[:3, 3] - approach_dist * tcp_pre_pose[:3, 2]
        tcp_final_pose = copy.deepcopy(tcp_pose)
        tcp_final_pose[:3, 3] = tcp_final_pose[:3, 3] + approach_dist / 3 * tcp_final_pose[:3, 2]
        self.movel(tcp_pre_pose, acc=acc, vel=vel)
        self.movel(tcp_pose, acc=acc, vel=vel)

    def grasp_and_throw(self, multifinger_grasp_used, two_fingers_grasp_used, cloud, multifinger_mesh_json_path, acc=0.05, vel=0.05, approach_dist=0.07, camera_pose=True,
                        execute_grasp=True, use_ready_pose=False, gripper_time=0.2):
        '''
        **Input:**
        - grasp: Grasp instance or numpy array of shape (4,4) or numpy array of shape (6,) in camera coordinate
        - acc: float of the maximum acceleration.
        - vel: float of the maximum velocity.
        - approach_dist: float of the distance to move along the z axis of tcp coordinate.
        - camera_pose: If true, grasp pose is given in camera coordinate. Else, it is given in tcp coordinate.
ss        **Output:**
        - No output but the robot moves to the ready pose first, and then moves to the given pose along the z axis of the tcp coordinate. Maybe it will close the gripper and move up to away pose and finally throw the object.
        '''
        self.angle = multifinger_grasp_used.angle
        multifinger_type = multifinger_grasp_used.grasp_type
        two_fingers_grasp_translation = two_fingers_grasp_used.translation
        two_fingers_grasp_rotation = two_fingers_grasp_used.rotation_matrix
        two_fingers_grasp_matrix = translation_rotation_2_matrix(two_fingers_grasp_translation, two_fingers_grasp_rotation)

        if isinstance(multifinger_grasp_used, Grasp) or isinstance(multifinger_grasp_used, InspireHandRGrasp) \
            or isinstance(multifinger_grasp_used, DH3Grasp) or isinstance(multifinger_grasp_used, AllegroGrasp):
            multifinger_translation = multifinger_grasp_used.translation
            multifinger_rotation = multifinger_grasp_used.rotation_matrix

            pose = translation_rotation_2_array(multifinger_translation, multifinger_rotation)
        elif isinstance(multifinger_grasp_used, np.ndarray):
            if multifinger_grasp_used.shape == (4, 4):
                pose = pose_matrix_2_array(multifinger_grasp_used)
            elif multifinger_grasp_used.shape == (6,):
                pose = multifinger_grasp_used
            else:
                raise ValueError('Shape of Grasp Array must be (4,4) or (6,), but it is {}'.format(multifinger_grasp_used.shape))
        else:
            raise ValueError('execute must be called with Grasp or numpy array, but it is {}'.format(type(multifinger_grasp_used)))
        pose = pose_array_2_matrix(pose)
        if camera_pose:
            tcp_pose = self.gripper_camera_pose_2_tcp_base_pose(pose, use_ready_pose=use_ready_pose)
        else:
            tcp_pose = copy.deepcopy(pose)
        target_gripper_pose = self.normalize(self.get_target_gripper_base_pose(two_fingers_grasp_matrix, use_ready_pose=use_ready_pose)[:3, 0])
        if self.gripper_type in ['InspireHandR', 'Allegro']:
            tcp_pose[:3, 3] = tcp_pose[:3, 3] + (multifinger_grasp_used.depth+0.014) * target_gripper_pose
        elif self.gripper_type == 'DH3':
            back_dis = self.angle[0] / 100 * 0.005
            tcp_pose[:3, 3] = tcp_pose[:3, 3] + (multifinger_grasp_used.depth - back_dis) * target_gripper_pose

        tcp_pre_pose = copy.deepcopy(tcp_pose)
        tcp_pre_pose[:3, 3] = tcp_pre_pose[:3, 3] - approach_dist * target_gripper_pose


        tcp_force_pose = copy.deepcopy(tcp_pose)
        tcp_force_pose[:3, 3] = tcp_force_pose[:3, 3] - 0.04 * tcp_force_pose[:3, 2]

        tcp_away_pose = copy.deepcopy(tcp_pose)

        # to avoid the gripper rotate around the z_{tcp} axis in the clock-wise direction.
        tcp_away_pose[:3, 3] = tcp_pre_pose[:3, 3] - (approach_dist + 0.05)* target_gripper_pose

        self.movels([tcp_pre_pose, tcp_pose], acc=acc, vel=vel, radius=0.02)
        data = dict()
        t1 = time.time()
        if execute_grasp:
            if self.gripper_type == 'Allegro':
                close_joint = np.array(grasp_types[str(int(multifinger_grasp_used.grasp_type))]['close_pose_matrix'])
                close_torque = np.array(grasp_types[str(int(multifinger_grasp_used.grasp_type))]['close_pose_torque'])
                self.close_gripper(close_joint, multifinger_grasp_used, close_torque)
            else:
                self.close_gripper()
            tcp_pre_pre_pose = copy.deepcopy(tcp_pose)
            tcp_pre_pre_pose[:3, 3] = tcp_pre_pre_pose[:3, 3] - 0.004 * target_gripper_pose
            self.movel(tcp_pre_pre_pose, acc=acc, vel=vel)
            time.sleep(0.1)
            self.movel(tcp_pre_pose, acc=acc, vel=vel)
            self.throw(acc=acc, vel=vel)
            self.open_gripper(multifinger_grasp_used.angle, sleep_time=gripper_time)
            if self.gripper_type == 'Allegro':
                self.set_torque(np.zeros(16))
        
        t2 = time.time()
        print(f'gripper time{t2 - t1}')

        camera_transform = self.get_cam2base_matrix() if self.global_cam else self.get_camera_tcp_matrix()
        return [tcp_pose, tcp_pre_pose, self.get_gripper_tcp_matrix(),\
               self.get_target_gripper_base_pose(two_fingers_grasp_matrix),  \
               camera_transform, self.get_tcp_base_matrix(use_ready_pose=use_ready_pose)]

    def get_tcp_base_matrix(self, use_ready_pose=False):
        '''
        **Output:**
        - Homogeneous transformation matrix of shape (4,4) for tcp(tool center point) to the base.
        '''
        if use_ready_pose:
            pose = np.array((self.ready_pose()), dtype=np.float32)
        else:
            pose = np.array((self.getl()), dtype=np.float32)
        return pose_array_2_matrix(pose)
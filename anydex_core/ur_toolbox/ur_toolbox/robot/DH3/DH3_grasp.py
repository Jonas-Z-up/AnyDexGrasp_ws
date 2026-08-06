import os
import json
import numpy as np
import open3d as o3d
import copy
import math

grasp_types = {'1':{'name': 'pose1', 'facenet_thumb': [60388], 'facenet_index': [69638, 51138], 'width':[0, 0.099]},
                '2':{'name': 'pose2', 'facenet_thumb': [60388], 'facenet_index': [69638, 51138], 'width':[0.007, 0.09]},
                '3':{'name': 'pose3', 'facenet_thumb': [60388], 'facenet_index': [69638, 51138], 'width':[0, 0.106]},
                '4':{'name': 'pose4', 'facenet_thumb': [60388], 'facenet_index': [69641, 51169], 'width':[0.006, 0.063]},
                }

MIN_GRASP_WIDTH = 0.01
MAX_GRASP_WIDTH = 0.099
DH3_DEFAULT_DEPTH = 0.0
DH3_ARRAY_LEN = 19

class DH3Grasp():
    def __init__(self, *args):
        '''
        **Input:**
        - args can be a numpy array or tuple of the score, width, height, depth, rotation_matrix, translation, object_id
        - the format of numpy array is [score, width, height, depth, rotation_matrix(9), translation(3), object_id]
        - the length of the numpy array is 17.
        '''
        if len(args) == 0:
            self.grasp_array = np.array(
                [0, 0.015, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 100, 0, 0.03], dtype=np.float64)
        elif len(args) == 1:
            if type(args[0]) == np.ndarray:
                self.grasp_array = copy.deepcopy(args[0])
            else:
                raise TypeError('if only one arg is given, it must be np.ndarray.')
        else:
            raise ValueError('only 1 or 7 arguments are accepted')

    def __repr__(self):
        return 'DH3Grasp: score:{}, depth:{}, grasp_type:{}, translation:{}\nrotation:\n{}\nobject_id:{}, angle:{}, width:{}'.format(
            self.score, self.depth, self.grasp_type, self.translation, self.rotation_matrix, self.angle, self.object_id, self.width)

    @property
    def score(self):
        '''
        **Output:**
        - float of the score.
        '''
        return self.grasp_array[0]

    @score.setter
    def score(self, score):
        '''
        **input:**
        - float of the score.
        '''
        self.grasp_array[0] = copy.deepcopy(score)

    @property
    def depth(self):
        '''
        **Output:**
        - float of the score.
        '''
        return self.grasp_array[1]

    @depth.setter
    def depth(self, depth):
        '''
        **input:**
        - float of the score.
        '''
        self.grasp_array[1] = copy.deepcopy(depth)

    @property
    def grasp_type(self):
        '''
        **Output:**
        - type of grasp.
        '''
        return self.grasp_array[2]

    @grasp_type.setter
    def grasp_type(self, grasp_type):
        '''
        **Input:**
        - type of grasp.
        '''
        self.grasp_array[2] = copy.deepcopy(grasp_type)

    @property
    def rotation_matrix(self):
        '''
        **Output:**
        - np.array of shape (3, 3) of the rotation matrix.
        '''
        return self.grasp_array[3:12].reshape((3,3))

    @rotation_matrix.setter
    def rotation_matrix(self, *args):
        '''
        **Input:**
        - len(args) == 1: tuple of matrix
        - len(args) == 9: float of matrix
        '''
        if len(args) == 1:
            self.grasp_array[3:12] = np.array(args[0],dtype = np.float64).reshape(9)
        elif len(args) == 9:
            self.grasp_array[3:12] = np.array(args,dtype = np.float64)

    @property
    def translation(self):
        '''
        **Output:**
        - np.array of shape (3,) of the translation.
        '''
        return self.grasp_array[12:15]

    @translation.setter
    def translation(self, *args):
        '''
        **Input:**
        - len(args) == 1: tuple of (x, y, z)
        - len(args) == 3: float of x, y, z
        '''
        if len(args) == 1:
            self.grasp_array[12:15] = np.array(args[0],dtype = np.float64)
        elif len(args) == 3:
            self.grasp_array[12:15] = np.array(args,dtype = np.float64)

    @property
    def angle(self):
        '''
        **Output:**
        - np.array of shape (2, ) of the angle.
        '''
        return self.grasp_array[15:17]

    @angle.setter
    def angle(self, *args):
        '''
        **Input:**
        - len(args) == 1: tuple of (position, rotation)
        - len(args) == 2: float of position, rotation
        '''
        if len(args) == 1:
            self.grasp_array[15:17] = np.array(args[0], dtype=np.float64)
        elif len(args) == 2:
            self.grasp_array[15:17] = np.array(args, dtype=np.float64)

    @property
    def object_id(self):
        '''
        **Output:**
        - float of the width.
        '''
        return float(self.grasp_array[17])

    @object_id.setter
    def object_id(self, object_id):
        '''
        **input:**
        - float of the width.
        '''
        self.grasp_array[17] = object_id

    @property
    def width(self):
        '''
        **Output:**
        - float of the width.
        '''
        return float(self.grasp_array[18])

    @width.setter
    def width(self, width):
        '''
        **input:**
        - float of the width.
        '''
        min_width, max_width = grasp_types[str(int(self.grasp_type))]['width']
        width = min(min(max_width, MAX_GRASP_WIDTH), max(max(min_width, MIN_GRASP_WIDTH), width))
        self.grasp_array[18] = width

    def get_array_grasp(self):
        '''
        **Output:**
        - numpy array of the grasp.
        '''
        return self.grasp_array

    def get_grasp_type_with_finger_name(self):
        '''
        **Output:**
        - name of the grasp type.
        '''
        return grasp_types[str(int(self.grasp_array[2]))]['name']

    def load_mesh_pointclouds(self, path_mesh, two_fingers_grasp, voxel_size=0.003):
        """Load DH3Grasp from dumped mesh_pointclouds.
        Args:
            path_mesh(str): the path that stores the mesh.
            grasp(graspnetAPI.Grasp)
            voxel_size:[0.001, 0.002, 0.003, 0.004]
        """
        source_mesh_pointclouds_path = os.path.join(path_mesh, 'meshes/source_pointclouds/voxel_size_' + str(int(voxel_size*1000)))
        return self._get_source_mesh_pointclouds_DH3(source_mesh_pointclouds_path, two_fingers_grasp)

    def _get_source_mesh_pointclouds_DH3(self, source_mesh_pointclouds_path, two_fingers_grasp):
        '''Load the source mesh
        **Input:**
            path_mesh(str): the path that stores the meshes.
            grasp(graspnetAPI.Grasp)
        **Output:**
            source_mesh: simplied DH3 mesh
        '''
        name = str(round(self.width * 100, 1)) + '.ply'
        finger_type = self.get_grasp_type_with_finger_name()
        source_mesh_pointclouds_path = os.path.join(source_mesh_pointclouds_path, finger_type, name)
        source_mesh_pointclouds = o3d.io.read_point_cloud(source_mesh_pointclouds_path)
        translation = self.translation.reshape(3, 1)
        rotation = self.rotation_matrix.reshape(3, 3)
        depth = self.depth

        t = two_fingers_grasp.translation.reshape(3, 1)
        r = two_fingers_grasp.rotation_matrix.reshape(3, 3)

        transform_mat_two_fingers = np.vstack((np.hstack((r, t)), np.array([0, 0, 0, 1])))
        grasp_direction = np.dot(transform_mat_two_fingers, np.array([[0.01], [0], [0], [1]]))[:3]
        grasp_direction = np.array(grasp_direction).reshape(3)
        grasp_direction = grasp_direction - two_fingers_grasp.translation.reshape((3))
        grasp_direction = self.normalize(grasp_direction)
        grasp_depth = np.array([[1, 0, 0, grasp_direction[0] * depth],
                                [0, 1, 0, grasp_direction[1] * depth],
                                [0, 0, 1, grasp_direction[2] * depth],
                                [0, 0, 0, 1]], dtype=np.float32)

        transform_mat = np.vstack((np.hstack((rotation, translation)), np.array([0, 0, 0, 1])))
        source_mesh_pointclouds.transform(transform_mat)
        source_mesh_pointclouds.transform(grasp_depth)
        return source_mesh_pointclouds

    def load_mesh(self, path_mesh, two_fingers_grasp):
        """Load DH3Grasp from dumped mesh.
        Args:
            path_mesh(str): the path that stores the mesh.
            grasp(graspnetAPI.Grasp)
        """
        source_mesh_path = os.path.join(path_mesh, 'meshes/source')
        return self._get_source_mesh_DH3(source_mesh_path, two_fingers_grasp)

    def _get_source_mesh_DH3(self, source_mesh_path, two_fingers_grasp):
        '''Load the source mesh
        **Input:**
            path_mesh(str): the path that stores the meshes.
            grasp(graspnetAPI.Grasp)
        **Output:**
            source_mesh: simplied DH3 mesh
        '''
        name = str(round(self.width * 100, 1)) + '.STL'
        finger_type = self.get_grasp_type_with_finger_name()
        source_mesh_path = os.path.join(source_mesh_path, finger_type, name)
        source_mesh = o3d.io.read_triangle_mesh(source_mesh_path)
        translation = self.translation.reshape(3, 1)
        rotation = self.rotation_matrix.reshape(3, 3)
        depth = self.depth

        t = two_fingers_grasp.translation.reshape(3, 1)
        r = two_fingers_grasp.rotation_matrix.reshape(3, 3)

        transform_mat_two_fingers = np.vstack((np.hstack((r, t)), np.array([0, 0, 0, 1])))
        grasp_direction = np.dot(transform_mat_two_fingers, np.array([[0.01], [0], [0], [1]]))[:3]
        grasp_direction = np.array(grasp_direction).reshape(3)
        grasp_direction = grasp_direction - two_fingers_grasp.translation.reshape((3))
        grasp_direction = self.normalize(grasp_direction)
        grasp_depth = np.array([[1, 0, 0, grasp_direction[0] * depth],
                                [0, 1, 0, grasp_direction[1] * depth],
                                [0, 0, 1, grasp_direction[2] * depth],
                                [0, 0, 0, 1]], dtype=np.float32)

        transform_mat = np.vstack((np.hstack((rotation, translation)), np.array([0, 0, 0, 1])))
        source_mesh.transform(transform_mat)
        source_mesh.transform(grasp_depth)
        return source_mesh

    def normalize(self, x):
        return np.array([x[0], x[1], x[2]]) / math.sqrt(np.power(x[0], 2) + np.power(x[1], 2) + np.power(x[2], 2))

    def _graspTR_2_DH3TR(self, two_fingers_grasp, path_json):
        '''
        **input:**
        - grasp: the pose of two-fingers
        **output:**
        - DH3_translation: the translations of the end of the robotic arm
        - DH3_rotation: the rotations of the end of the robotic arm
        - DH3_angle: the angle of DH3
        '''
        width = two_fingers_grasp.width
        min_width, max_width = grasp_types[str(self.grasp_type)]['width']
        width = min(min(max_width, MAX_GRASP_WIDTH), max(max(min_width, MIN_GRASP_WIDTH), width))
        translation = two_fingers_grasp.translation
        rotation = two_fingers_grasp.rotation_matrix

        mesh_information_path = os.path.join(path_json, 'width_12D_angle_2D_angle.json')
        with open(mesh_information_path, 'r', encoding='UTF-8') as f:
            width_12D_angle_2D_angle = json.load(f)

        matrix_two_fingers = np.vstack(
            (np.hstack((rotation, np.array(translation).reshape((3, 1)))), np.array((0, 0, 0, 1))))
        translation_DH3 = width_12D_angle_2D_angle[self.get_grasp_type_with_finger_name()][str(np.round(width * 100, 1))]['translation']
        rotation_DH3 = width_12D_angle_2D_angle[self.get_grasp_type_with_finger_name()][str(np.round(width * 100, 1))]['rotation']
        matrix_DH3 = np.vstack(
            (np.hstack((rotation_DH3, np.array(translation_DH3).reshape((3, 1)))), np.array((0, 0, 0, 1))))
        mat_two_fingers_2_DH3 = np.dot(matrix_two_fingers, np.linalg.inv(matrix_DH3))
        DH3_translation = np.array([mat_two_fingers_2_DH3[0,3], mat_two_fingers_2_DH3[1,3], mat_two_fingers_2_DH3[2,3]])
        DH3_rotation = np.array([mat_two_fingers_2_DH3[0,0:3], mat_two_fingers_2_DH3[1,0:3], mat_two_fingers_2_DH3[2,0:3]])

        angle = width_12D_angle_2D_angle[self.get_grasp_type_with_finger_name()][str(np.round(width * 100, 1))]['2d']
        self.width = width
        self.translation = DH3_translation
        self.rotation_matrix = DH3_rotation
        self.angle = angle
        # return DH3_translation, DH3_rotation, angle

    def from_grasp(self, two_fingers_grasp, DH3type, path_json):
        """Grasp to DH3Grasp Transformation.
        Args:
            grasp(graspnetAPI.Grasp): the grasp to be transformed.
            DH3type(np.array(int)): the types of ThreeFingersGrasp.
        """
        self.score = two_fingers_grasp.score
        self.depth = two_fingers_grasp.depth
        self.grasp_type = DH3type
        self.width = two_fingers_grasp.width
        self._graspTR_2_DH3TR(two_fingers_grasp, path_json)
        self.object_id = two_fingers_grasp.object_id
        

    def from_numpy(self, npy_file_path):
        '''
        **Input:**
        - npy_file_path: string of the file path.
        '''
        self.grasp_array = np.load(npy_file_path)
        return self


class DH3GraspGroup():
    def __init__(self, *args):
        '''
        **Input:**
        - args can be (1) nothing (2) numpy array of grasp group array (3) str of the npy file.
        '''
        if len(args) == 0:
            self.grasp_group_array = np.zeros((0, DH3_ARRAY_LEN), dtype=np.float64)
        elif len(args) == 1:
            if isinstance(args[0], np.ndarray):
                self.grasp_group_array = args[0]
            elif isinstance(args[0], str):
                self.grasp_group_array = np.load(args[0])
            else:
                raise ValueError('args must be nothing, numpy array or string.')
        else:
            raise ValueError('args must be nothing, numpy array or string.')

    def __len__(self):
        '''
        **Output:**
        - int of the length.
        '''
        return len(self.grasp_group_array)

    def __repr__(self):
        repr = '----------\nDH3 Grasp Group, Number={}:\n'.format(self.__len__())
        if self.__len__() <= 6:
            for grasp_array in self.grasp_group_array:
                repr += DH3Grasp(grasp_array).__repr__() + '\n'
        else:
            for i in range(3):
                repr += DH3Grasp(self.grasp_group_array[i]).__repr__() + '\n'
            repr += '......\n'
            for i in range(3):
                repr += DH3Grasp(self.grasp_group_array[-(3 - i)]).__repr__() + '\n'
        return repr + '----------'

    def __getitem__(self, index):
        '''
        **Input:**
        - index: int, slice, list or np.ndarray.
        **Output:**
        - if index is int, return Grasp instance.
        - if index is slice, np.ndarray or list, return GraspGroup instance.
        '''
        if type(index) == int:
            return DH3Grasp(self.grasp_group_array[index])
        elif type(index) == slice:
            graspgroup = DH3GraspGroup()
            graspgroup.grasp_group_array = copy.deepcopy(self.grasp_group_array[index])
            return graspgroup
        elif type(index) == np.ndarray:
            return DH3GraspGroup(self.grasp_group_array[index])
        elif type(index) == list:
            return DH3GraspGroup(self.grasp_group_array[index])
        else:
            raise TypeError('unknown type "{}" for calling __getitem__ for DH3GraspGroup'.format(type(index)))

    @property
    def scores(self):
        '''
        **Output:**
        - numpy array of shape (-1, ) of the scores.
        '''
        return self.grasp_group_array[:, 0]

    @scores.setter
    def scores(self, scores):
        '''
        **Input:**
        - scores: numpy array of shape (-1, ) of the scores.
        '''
        assert scores.size == len(self)
        self.grasp_group_array[:, 0] = copy.deepcopy(scores)

    @property
    def depths(self):
        '''
        **Output:**
        - numpy array of shape (-1, ) of the scores.
        '''
        return self.grasp_group_array[:, 1]

    @depths.setter
    def depths(self, depths):
        '''
        **Input:**
        - scores: numpy array of shape (-1, ) of the scores.
        '''
        assert depths.size == len(self)
        self.grasp_group_array[:, 1] = copy.deepcopy(depths)

    @property
    def grasp_types(self):
        '''
        **Output:**
        - numpy array of shape (-1, ) of the grasp types.
        '''
        return self.grasp_group_array[:, 2]

    @grasp_types.setter
    def grasp_types(self, grasp_types):
        '''
        **Input:**
        - grasp_types: numpy array of shape (-1, ) of the grasp types.
        '''
        assert grasp_types.size == len(self)
        self.grasp_group_array[:, 2] = copy.deepcopy(grasp_types)

    @property
    def rotation_matrices(self):
        '''
        **Output:**
        - np.array of shape (-1, 3, 3) of the rotation matrices.
        '''
        return self.grasp_group_array[:, 3:12].reshape((-1, 3, 3))

    @rotation_matrices.setter
    def rotation_matrices(self, rotation_matrices):
        '''
        **Input:**
        - rotation_matrices: numpy array of shape (-1, 3, 3) of the rotation_matrices.
        '''
        assert rotation_matrices.shape == (len(self), 3, 3)
        self.grasp_group_array[:, 3:12] = copy.deepcopy(rotation_matrices.reshape((-1, 9)))

    @property
    def translations(self):
        '''
        **Output:**
        - np.array of shape (-1, 3) of the translations.
        '''
        return self.grasp_group_array[:, 12:15]

    @translations.setter
    def translations(self, translations):
        '''
        **Input:**
        - translations: numpy array of shape (-1, 3) of the translations.
        '''
        assert translations.shape == (len(self), 3)
        self.grasp_group_array[:, 12:15] = copy.deepcopy(translations)

    @property
    def angles(self):
        '''
        **Output:**
        - np.array of shape (-1, 2) of the angles.
        '''
        return self.grasp_group_array[:, 15:17]

    @angles.setter
    def angles(self, angles):
        '''
        **Input:**
        - angles: numpy array of shape (-1, 2) of the angles.
        '''
        assert angles.shape == (len(self), 2)
        self.grasp_group_array[:, 15:17] = copy.deepcopy(angles)

    @property
    def object_ids(self):
        '''
        **Output:**
        - numpy array of shape (-1, ) of the object_ids.
        '''
        return self.grasp_group_array[:, 17]

    @object_ids.setter
    def object_ids(self, object_ids):
        '''
        **Input:**
        - object_ids: numpy array of shape (-1, ) of the object_ids.
        '''
        assert object_ids.size == len(self)
        self.grasp_group_array[:, 17] = copy.deepcopy(object_ids)

    @property
    def widths(self):
        '''
        **Output:**
        - numpy array of shape (-1, ) of the widths.
        '''
        return self.grasp_group_array[:, 18]

    @widths.setter
    def widths(self, widths):
        '''
        **Input:**
        - widths: numpy array of shape (-1, ) of the widths.
        '''
        assert widths.size == len(self)
        for ids, width in enumerate(widths):
            min_width, max_width = grasp_types[str(int(self.grasp_types[ids]))]['width']
            widths[ids] = min(min(max_width, MAX_GRASP_WIDTH), max(max(min_width, MIN_GRASP_WIDTH), widths[ids]))
        self.grasp_group_array[:, 18] = copy.deepcopy(widths)

    def set_grasp_min_width(self, MIN_GRASP_WIDTH):
        '''
        - set min width of the grasp.
        '''
        MIN_GRASP_WIDTH = MIN_GRASP_WIDTH

    def get_graspgroup_types_with_finger_names(self):
        '''
        **Output:**
        - names of the graspgroup type.
        '''
        graspgroup_types_with_finger_names = []
        for finger_type in self.grasp_group_array[:, 2]:
            graspgroup_types_with_finger_names.append(grasp_types[str(int(finger_type))]['name'])
        return graspgroup_types_with_finger_names

    def graspgroupTR_2_TR(self, graspgroup, path_json):
        '''
        **input:**
        - graspgroup: the pose of two-fingers
        **output:**
        - DH3_translations: the translations of the end of the robotic arm
        - DH3_rotations: the rotations of the end of the robotic arm
        - DH3_angles: the angle of DH3
        '''
        widths = graspgroup.widths
        translations = graspgroup.translations
        rotations = graspgroup.rotation_matrices

        mesh_information_path = os.path.join(path_json, 'width_12D_angle_2D_angle.json')
        with open(mesh_information_path, 'r', encoding='UTF-8') as f:
            width_12D_angle_2D_angle = json.load(f)

        DH3_translations = []
        DH3_rotations = []
        DH3_angles = []
        DH3_widths = []
        for idx, width in enumerate(widths):
            min_width, max_width = grasp_types[str(int(self.grasp_types[idx]))]['width']
            width = min(min(max_width, MAX_GRASP_WIDTH), max(max(min_width, MIN_GRASP_WIDTH), width))
            translation_two_fingers = translations[idx]
            rotation_two_fingers = rotations[idx]
            DH3_widths.append(width)
            matrix_two_fingers = np.vstack(
                (np.hstack((rotation_two_fingers, np.array(translation_two_fingers).reshape((3, 1)))), np.array((0, 0, 0, 1))))
            translation_DH3 = width_12D_angle_2D_angle[self.get_graspgroup_types_with_finger_names()[idx]][str(np.round(width * 100, 1))]['translation']
            rotation_DH3 = width_12D_angle_2D_angle[self.get_graspgroup_types_with_finger_names()[idx]][str(np.round(width * 100, 1))]['rotation']
            matrix_DH3 = np.vstack(
                (np.hstack((rotation_DH3, np.array(translation_DH3).reshape((3, 1)))), np.array((0, 0, 0, 1))))
            mat_two_fingers_2_DH3 = np.dot(matrix_two_fingers, np.linalg.inv(matrix_DH3))
            DH3_translations.append([mat_two_fingers_2_DH3[0,3], mat_two_fingers_2_DH3[1,3], mat_two_fingers_2_DH3[2,3]])
            rotation = np.array([mat_two_fingers_2_DH3[0,0:3], mat_two_fingers_2_DH3[1,0:3], mat_two_fingers_2_DH3[2,0:3]])
            DH3_rotations.append(rotation)

            angle = width_12D_angle_2D_angle[self.get_graspgroup_types_with_finger_names()[idx]][str(np.round(width * 100, 1))]['2d']
            DH3_angles.append(angle)
        self.widths = np.array(DH3_widths)
        self.translations = np.array(DH3_translations)
        self.rotation_matrices = np.array(DH3_rotations)
        self.angles = np.array(DH3_angles)

    def from_npy(self, npy_file_path):
        '''
        **Input:**
        - npy_file_path: string of the file path.
        '''
        self.grasp_group_array = np.load(npy_file_path)
        return self

    def from_graspgroup(self, graspgroup, DH3types, path_json):
        """Grasp to ThreeFingersGraspGroup Transformation.
        Args:
            graspgroup(graspnetAPI.GraspGroup): the graspgroup to be transformed.
            ffgtypes(np.array(int)): the types of ThreeFingersGrasp.
        """
        self.grasp_group_array = np.zeros((len(graspgroup), DH3_ARRAY_LEN), dtype=np.float64)
        self.scores = graspgroup.scores
        self.depths = graspgroup.depths
        self.grasp_types = DH3types
        self.widths = graspgroup.widths
        self.graspgroupTR_2_TR(graspgroup, path_json)
        self.object_ids = graspgroup.object_ids
        return self

    def load_meshes_pointclouds(self, path_mesh, two_fingers_grasp, voxel_size=0.003):
        """Load DH3Grasp from dumped mesh_pointclouds.
        Args:
            path_mesh(str): the path that stores the mesh.
            grasp(graspnetAPI.Grasp)
            voxel_size:[0.001, 0.002, 0.003, 0.004]
        """
        source_meshes_pointclouds_path = os.path.join(path_mesh, 'meshes/source_pointclouds/voxel_size_' + str(int(voxel_size*1000)))
        return self._get_source_meshes_pointclouds_DH3(source_meshes_pointclouds_path, two_fingers_grasp)

    def _get_source_meshes_pointclouds_DH3(self, source_meshes_pointclouds_path, two_fingers_ggarray):
        '''Load the source mesh
        **Input:**
            path_mesh(str): the path that stores the meshes.
            grasp(graspnetAPI.Grasp)
        **Output:**
            source_mesh: simplied DH3 mesh
        '''
        source_meshes_pointclouds_DH3 = []
        for id in range(self.__len__()):
            min_width, max_width = grasp_types[str(int(self.grasp_types[id]))]['width']
            width = min(min(max_width, MAX_GRASP_WIDTH), max(max(min_width, MIN_GRASP_WIDTH), self.widths[id]))
            name = str(round(width * 100, 1)) + '.ply'
            finger_type = self.get_graspgroup_types_with_finger_names()[id]
            source_mesh_pointclouds_path = os.path.join(source_meshes_pointclouds_path, finger_type, name)
            source_mesh_pointclouds = o3d.io.read_point_cloud(source_mesh_pointclouds_path)

            translation = self.translations[id].reshape(3, 1)
            rotation = self.rotation_matrices[id].reshape(3, 3)
            depth = self.depths[id]

            t = two_fingers_ggarray.translations[id].reshape(3, 1)
            r = two_fingers_ggarray.rotation_matrices[id].reshape(3, 3)

            transform_mat_two_fingers = np.vstack((np.hstack((r, t)), np.array([0, 0, 0, 1])))

            grasp_direction = np.dot(transform_mat_two_fingers, np.array([[0.01], [0], [0], [1]]))[:3]
            grasp_direction = np.array(grasp_direction).reshape(3)
            grasp_direction = grasp_direction - two_fingers_ggarray.translations[id].reshape((3))
            grasp_direction = self.normalize(grasp_direction)
            grasp_depth = np.array([[1, 0, 0, grasp_direction[0] * depth],
                                    [0, 1, 0, grasp_direction[1] * depth],
                                    [0, 0, 1, grasp_direction[2] * depth],
                                    [0, 0, 0, 1]], dtype=np.float32)

            transform_mat = np.vstack((np.hstack((rotation, translation)), np.array([0, 0, 0, 1])))
            source_mesh_pointclouds.transform(transform_mat)
            source_mesh_pointclouds.transform(grasp_depth)
            source_meshes_pointclouds_DH3.append(source_mesh_pointclouds)
        return np.array(source_meshes_pointclouds_DH3)

    def load_meshes(self, path_mesh, two_fingers_ggarray):
        """Load DH3GraspGroup from dumped meshes.
        Args:
            path_mesh(str): the path that stores the meshes.
            graspgroup(graspnetAPI.GraspGroup)
        """
        source_meshes_path = os.path.join(path_mesh, 'meshes/source')
        return self._get_source_meshes_DH3(source_meshes_path, two_fingers_ggarray)

    def _get_source_meshes_DH3(self, source_meshes_path, two_fingers_ggarray):
        '''Load the source meshes
        **Input:**
            path_mesh(str): the path that stores the meshes.
            graspgroup(graspnetAPI.GraspGroup)
        **Output:**
            source_meshes_DH3: simplied DH3 meshes
        '''
        source_meshes_DH3  = []
        for id in range(self.__len__()):
            min_width, max_width = grasp_types[str(int(self.grasp_types[id]))]['width']
            widths[ids] = min(min(max_width, MAX_GRASP_WIDTH), max(max(min_width, MIN_GRASP_WIDTH), self.widths[id]))
            name = str(round(width * 100, 1)) + '.STL'
            finger_type = self.get_graspgroup_types_with_finger_names()[id]
            source_mesh_path = os.path.join(source_meshes_path, finger_type, name)
            source_mesh = o3d.io.read_triangle_mesh(source_mesh_path)

            translation = self.translations[id].reshape(3, 1)
            rotation = self.rotation_matrices[id].reshape(3, 3)
            depth = self.depths[id]

            t = two_fingers_ggarray.translations[id].reshape(3, 1)
            r = two_fingers_ggarray.rotation_matrices[id].reshape(3, 3)

            transform_mat_two_fingers = np.vstack((np.hstack((r, t)), np.array([0, 0, 0, 1])))

            grasp_direction = np.dot(transform_mat_two_fingers, np.array([[0.01], [0], [0], [1]]))[:3]
            grasp_direction = np.array(grasp_direction).reshape(3)
            grasp_direction = grasp_direction - two_fingers_ggarray.translations[id].reshape((3))
            grasp_direction = self.normalize(grasp_direction)
            grasp_depth = np.array([[1, 0, 0, grasp_direction[0] * depth],
                                    [0, 1, 0, grasp_direction[1] * depth],
                                    [0, 0, 1, grasp_direction[2] * depth],
                                    [0, 0, 0, 1]], dtype=np.float32)

            transform_mat = np.vstack((np.hstack((rotation, translation)), np.array([0, 0, 0, 1])))
            source_mesh.transform(transform_mat)
            source_mesh.transform(np.array(grasp_depth))
            source_meshes_DH3.append(source_mesh)
        return np.array(source_meshes_DH3)

    def normalize(self, x):
        return np.array([x[0], x[1], x[2]]) / math.sqrt(np.power(x[0], 2) + np.power(x[1], 2) + np.power(x[2], 2))

    def sort_by_score(self, reverse=False):
        '''
        **Input:**
        - reverse: bool of order, if False, from high to low, if True, from low to high.
        **Output:**
        - no output but sort the grasp group.
        '''
        score = self.grasp_group_array[:, 0]
        index = np.argsort(score)
        if not reverse:
            index = index[::-1]
        self.update_by_id(index)
        return self

    def filter_grasp_group_by_z_axis(self, angle_with_z_axis):
        '''Filter the graspgroup by angle with z axis and the worksapce
        **input:**
            angle_with_z_axis(float32): [0, 1]
        '''
        mask = (self.grasp_group_array[:,9] > angle_with_z_axis) & (self.grasp_group_array[:,18] < MAX_GRASP_WIDTH)
        workspace_mask = (self.grasp_group_array[:,12] > -0.23) & (self.grasp_group_array[:,12] < 0.23) & \
                         (self.grasp_group_array[:,13] > -0.12) & (self.grasp_group_array[:,13] < 0.17)
        index = mask & workspace_mask
        self.update_by_id(index)
        return index

    def update_by_id(self, index):
        '''Modify the order of the grasp_group_array by index
        **input:**
            index: numpy array of shape (len(grasp_group_array), 1).
        '''
        self.grasp_group_array = self.grasp_group_array[index]
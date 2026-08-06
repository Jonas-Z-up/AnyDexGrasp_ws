import copy
import os
import json
import math
import sys
import numpy as np
from tqdm import tqdm
import open3d as o3d
from transforms3d.euler import euler2mat, quat2mat
import pybullet as p
import pybullet_data

from graspnetAPI import Grasp


class InspireAngles2STLs():
    def __init__(self, grasp_types):
        self.grasp_types = grasp_types
    
    def normalize(self, x):
        return np.array([x[0], x[1], x[2]]) / math.sqrt(np.power(x[0], 2) + np.power(x[1], 2) + np.power(x[2], 2))

    def get_normal_vector(self, p1, p2, p3):
        a = ((p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]))
        b = ((p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2]))
        c = ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
        return np.array([a, b, c])

    def di(self, x, y):
        return np.sqrt(np.power(x[0] - y[0], 2) + np.power(x[1] - y[1], 2) + np.power(x[2] - y[2], 2))

    def compute_center(self, meshes, facenet_thumb, facenet_index):
        distance1s = []
        triangle1 = meshes.triangles[facenet_thumb[0]]
        distance = [meshes.vertices[triangle1[0]], meshes.vertices[triangle1[1]], meshes.vertices[triangle1[2]]]
        distance1s.append(distance)

        triangle1 = meshes.triangles[facenet_thumb[1]]
        distance = [meshes.vertices[triangle1[0]], meshes.vertices[triangle1[1]], meshes.vertices[triangle1[2]]]
        distance1s.append(distance)

        distance2s = []
        triangle1 = meshes.triangles[facenet_index[0]]
        distance = [meshes.vertices[triangle1[0]], meshes.vertices[triangle1[1]], meshes.vertices[triangle1[2]]]
        distance2s.append(distance)
        triangle1 = meshes.triangles[facenet_index[1]]

        distance = [meshes.vertices[triangle1[0]], meshes.vertices[triangle1[1]], meshes.vertices[triangle1[2]]]
        distance2s.append(distance)

        midpoint = []
        center1 = []
        center2 = []
        for n in range(2):
            center1.append((distance1s[n][0] + distance1s[n][1] + distance1s[n][2]) / 3.0)
            center2.append((distance2s[n][0] + distance2s[n][1] + distance2s[n][2]) / 3.0)
        center = [(center1[0] + center1[1]) / 2.0, (center2[0] + center2[1])/ 2.0]
        midpoint = (center[0] + center[1]) / 2.0
        mesh_di = self.di(center[0], center[1])
        return midpoint, center, mesh_di

    def get_center_orientation(self, meshes, facnet):
        triangle1 = meshes.triangles[facnet]
        point = (meshes.vertices[triangle1[0]] + meshes.vertices[triangle1[1]] + meshes.vertices[triangle1[2]]) / 3
        return point

    def thumb_index_grasp_inforamtion(self, meshes, facenet_thumb, facenet_index):
        point0 = (self.get_center_orientation(meshes, facenet_thumb[0][0]) + self.get_center_orientation(meshes, facenet_thumb[0][1])) / 2
        point1 = (self.get_center_orientation(meshes, facenet_index[0][0]) + self.get_center_orientation(meshes, facenet_index[0][1]))/2
        grasp_translation = (point0 + point1) / 2

        tool_x = self.normalize(self.get_normal_vector(point0, point1, self.get_center_orientation(meshes, facenet_index[0][1])))
        tool_y = self.normalize(point0 - grasp_translation)
        tool_z = np.cross(tool_x, tool_y)
        grasp_rotation = np.c_[tool_x, tool_y, tool_z]
        return grasp_translation, grasp_rotation 

    def common_grasp_information(self, grasp_type, meshes, facenet_thumb, facenet_index):
        midpoint1, centers1, mesh_di1 = self.compute_center(meshes, facenet_thumb=facenet_thumb[0],
                                                                    facenet_index=facenet_index[0])
        midpoint2, centers2, mesh_di2 = self.compute_center(meshes, facenet_thumb=facenet_thumb[0],
                                                        facenet_index=facenet_index[1])
        midpoint = (midpoint1 + midpoint2) / 2.0
        centers = [(centers1[0] + centers2[0]) / 2.0, (centers1[1] + centers2[1]) / 2.0]


        triangle1 = meshes.triangles[145998]
        center1 = (meshes.vertices[triangle1[0]] + meshes.vertices[triangle1[1]] + meshes.vertices[triangle1[2]]) / 3.0
        triangle2 = meshes.triangles[145999]
        center2 = (meshes.vertices[triangle2[0]] + meshes.vertices[triangle2[1]] + meshes.vertices[triangle2[2]]) / 3.0
        x1 = (center1 + center2) / 2.0

        triangle1 = meshes.triangles[52980]
        center1 = (meshes.vertices[triangle1[0]] + meshes.vertices[triangle1[1]] + meshes.vertices[triangle1[2]]) / 3.0
        triangle2 = meshes.triangles[52981]
        center2 = (meshes.vertices[triangle2[0]] + meshes.vertices[triangle2[1]] + meshes.vertices[triangle2[2]]) / 3.0
        x2 = (center1 + center2) / 2.0

        x = (x1+x2)/2.0

        if grasp_type == 0:
            x = x1
            midpoint = midpoint1
            centers = centers1
        vector = self.get_normal_vector(centers[0], centers[1], x)
        vector = self.normalize(vector) / 40
        p4 = np.array(midpoint) + np.array(vector)
        normal_vector = self.get_normal_vector(centers[1], centers[0], p4)
        normal_vector = self.normalize(normal_vector) / 40
        p5 = np.array(midpoint) + np.array(normal_vector)

        rotation = [[list(self.normalize(p5 - midpoint))[0], list(self.normalize(centers[0] - midpoint))[0], list(self.normalize(p4 - midpoint))[0]],
                    [list(self.normalize(p5 - midpoint))[1], list(self.normalize(centers[0] - midpoint))[1], list(self.normalize(p4 - midpoint))[1]],
                    [list(self.normalize(p5 - midpoint))[2], list(self.normalize(centers[0] - midpoint))[2], list(self.normalize(p4 - midpoint))[2]]]
        
        translation = midpoint
        return np.array(translation), np.array(rotation)
    
    def special_grasp_information(self, grasp_type, meshes, facenet_thumb, facenet_index):
        point0 = self.get_center_orientation(meshes, facenet_thumb[0][0])
        point1 = (self.get_center_orientation(meshes, facenet_index[0][0]) + self.get_center_orientation(meshes, facenet_index[0][1]))/2
        point2 = self.get_center_orientation(meshes, facenet_index[1][0])

        end_point0 = point0
        end_point1 = (point1 + point2) / 2
        grasp_translation = (end_point0 + end_point1) / 2

        # The finger is not vertical
        if grasp_type == 10:
            grasp_translation[1] = grasp_translation[1] - 0.003
        tool_x = self.normalize(self.get_normal_vector(point0, point1, point2))
        tool_y = self.normalize(point0 - grasp_translation)
        tool_z = np.cross(tool_x, tool_y)
        grasp_rotation = np.c_[tool_x, tool_y, tool_z]
        return grasp_translation, grasp_rotation

    def thumb_index_lateral_grasp_information(self, meshes, facenet_thumb, facenet_index):
        point0 = self.get_center_orientation(meshes, facenet_thumb[0][0])
        point1 = (self.get_center_orientation(meshes, facenet_index[0][0]) + self.get_center_orientation(meshes, facenet_index[0][1]))/2
        grasp_translation = (point0 + point1) / 2

        tool_x = self.normalize(self.get_normal_vector(point1, point0, self.get_center_orientation(meshes, facenet_index[0][1])))
        tool_y = self.normalize(point0 - grasp_translation)
        tool_z = np.cross(tool_x, tool_y)
        grasp_rotation = np.c_[tool_x, tool_y, tool_z]
        return grasp_translation, grasp_rotation

    def get_pose_information(self, meshes, grasp_type, vis=False):
        facenet_thumb = self.grasp_types[str(grasp_type+1)]['facenet_thumb']
        facenet_index = self.grasp_types[str(grasp_type+1)]['facenet_index']
        if grasp_type < 4:
            translation, rotation = self.common_grasp_information(grasp_type, meshes, facenet_thumb, facenet_index)
        elif grasp_type < 11:
            translation, rotation = self.special_grasp_information(grasp_type, meshes, facenet_thumb, facenet_index)
        elif grasp_type == 11:
            translation, rotation = self.thumb_index_lateral_grasp_information(meshes, facenet_thumb, facenet_index)

        if vis:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            grasp_matrix = np.vstack((np.hstack((rotation, translation.reshape((3, 1)))), np.array((0, 0, 0, 1))))
            frame_grasp = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1).transform(grasp_matrix)
            o3d.visualization.draw_geometries([meshes, frame, frame_grasp])
        return translation, rotation


class DH3Angles2STLs():
    def __init__(self, grasp_types):
        self.grasp_types = grasp_types
    
    def normalize(self, x):
        return np.array([x[0], x[1], x[2]]) / math.sqrt(np.power(x[0], 2) + np.power(x[1], 2) + np.power(x[2], 2))

    def get_normal_vector(self, p1, p2, p3):
        a = ((p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]))
        b = ((p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2]))
        c = ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
        return np.array([a, b, c])

    def di(self, x, y):
        return np.sqrt(np.power(x[0] - y[0], 2) + np.power(x[1] - y[1], 2) + np.power(x[2] - y[2], 2))

    def get_center_orientation(self, meshes, facnet):
        triangle1 = meshes.triangles[facnet]
        point = (meshes.vertices[triangle1[0]] + meshes.vertices[triangle1[2]]) / 2
        return point
    
    def common_grasp_information(self, meshes, facenet_thumb, facenet_index):
        point0 = self.get_center_orientation(meshes, facenet_thumb[0])
        point1 = self.get_center_orientation(meshes, facenet_index[0])
        point2 = self.get_center_orientation(meshes, facenet_index[1])

        end_point0 = point0
        end_point1 = (point1 + point2) / 2
        grasp_translation = (end_point0 + end_point1) / 2

        tool_x = self.normalize(self.get_normal_vector(point1, point0, point2))
        tool_y = self.normalize(point0 - grasp_translation)
        tool_z = np.cross(tool_x, tool_y)
        grasp_rotation = np.c_[tool_x, tool_y, tool_z]
        return grasp_translation, grasp_rotation

    def pose2_information(self, meshes, facenet_thumb, facenet_index):
        point0 = self.get_center_orientation(meshes, facenet_thumb[0])
        point1 = self.get_center_orientation(meshes, facenet_index[0])
        point2 = self.get_center_orientation(meshes, facenet_index[1])

        grasp_translation = (point1 + point2) / 2

        tool_x = self.normalize(self.get_normal_vector(point1, point0, point2))
        tool_y = self.normalize(point2 - grasp_translation)
        tool_z = np.cross(tool_x, tool_y)
        grasp_rotation = np.c_[tool_x, tool_y, tool_z]
        return grasp_translation, grasp_rotation

    def pose3_information(self, meshes, facenet_thumb, facenet_index):
        point0 = self.get_center_orientation(meshes, facenet_thumb[0])
        point1 = meshes.vertices[meshes.triangles[facenet_index[0]][2]]
        point2 = meshes.vertices[meshes.triangles[facenet_index[1]][0]]

        end_point0 = point0
        end_point1 = (point1 + point2) / 2
        grasp_translation = (end_point0 + end_point1) / 2

        tool_x = self.normalize(self.get_normal_vector(point1, point0, point2))
        tool_y = self.normalize(point0 - grasp_translation)
        tool_z = np.cross(tool_x, tool_y)
        grasp_rotation = np.c_[tool_x, tool_y, tool_z]
        return grasp_translation, grasp_rotation

    def get_pose_information(self, meshes, grasp_type, vis=False):
        facenet_thumb = self.grasp_types[str(grasp_type+1)]['facenet_thumb']
        facenet_index = self.grasp_types[str(grasp_type+1)]['facenet_index']
        if grasp_type < 2:
            translation, rotation = self.common_grasp_information(meshes, facenet_thumb, facenet_index)
        elif grasp_type == 2:
            translation, rotation = self.pose2_information(meshes, facenet_thumb, facenet_index)
        elif grasp_type == 3:
            translation, rotation = self.pose3_information(meshes, facenet_thumb, facenet_index)

        if vis:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            grasp_matrix = np.vstack((np.hstack((rotation, translation.reshape((3, 1)))), np.array((0, 0, 0, 1))))
            frame_grasp = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1).transform(grasp_matrix)
            o3d.visualization.draw_geometries([meshes, frame, frame_grasp])
        return translation, rotation


class AllegroAngles2STLs():
    def __init__(self, grasp_types):
        self.grasp_types = grasp_types
    
    def normalize(self, x):
        return np.array([x[0], x[1], x[2]]) / math.sqrt(np.power(x[0], 2) + np.power(x[1], 2) + np.power(x[2], 2))

    def get_normal_vector(self, p1, p2, p3):
        a = ((p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]))
        b = ((p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2]))
        c = ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
        return np.array([a, b, c])

    def di(self, x, y):
        return np.sqrt(np.power(x[0] - y[0], 2) + np.power(x[1] - y[1], 2) + np.power(x[2] - y[2], 2))

    def get_center_orientation(self, meshes, facenet):
        facenet_id, triangle_id = facenet
        triangle1 = meshes.triangles[facenet_id]
        point = meshes.vertices[triangle1[triangle_id]]
        return point
    
    def get_coordinate_origin(self, meshes, facnet):
        triangle1 = meshes.triangles[facnet]
        point = (meshes.vertices[triangle1[0]] + meshes.vertices[triangle1[2]]) / 2
        return point

    def common_grasp_information(self, meshes, facenet_thumb, facenet_index, width, grasp_type):
        point0 = [0, 0, 0]
        for i in range(len(facenet_thumb)):
            point0 = point0 + self.get_center_orientation(meshes, facenet_thumb[i])
        point0 = point0 / len(facenet_thumb)
        
        point1 = self.get_center_orientation(meshes, facenet_index[0])
        point2 = self.get_center_orientation(meshes, facenet_index[1])

        end_point0 = point0
        end_point1 = (point1 + point2) / 2

        grasp_translation = (end_point0 + end_point1) / 2

        tool_x = self.normalize(self.get_normal_vector(point0, point1, point2))
        tool_y = self.normalize(point0 - grasp_translation)
        if grasp_type in [5]:
            end_point0 = point0
            end_point1 = point1
            grasp_translation = (end_point0 + end_point1) / 2

            tool_x = grasp_translation - (self.get_center_orientation(meshes, facenet_index[2]) + self.get_center_orientation(meshes, facenet_index[1])) / 2
            tool_y = self.get_center_orientation(meshes, facenet_index[0]) - self.get_center_orientation(meshes, facenet_thumb[0])
            tool_x = self.normalize(tool_x)
            tool_y = self.normalize(tool_y)

        tool_z = np.cross(tool_x, tool_y)
        grasp_rotation = np.c_[tool_x, tool_y, tool_z]
        if grasp_type in [11]:
            grasp_rotation = np.c_[-tool_z, tool_y, tool_x]
  
        return grasp_translation, grasp_rotation

    def special_grasp_information(self, meshes, facenet_index, width, grasp_type):
        point0 = self.get_center_orientation(meshes, facenet_index[0])
        point1 = self.get_center_orientation(meshes, facenet_index[1])
        point2 = self.get_center_orientation(meshes, facenet_index[2])

        end_point0 = (point0 + point1) / 2 - [0, 0, 0.015] - 0.00014 * width * np.array([0, 0, 1])
        grasp_translation = end_point0 + width * 0.01 * np.array([0, 0, -1.2]) / 2 

        tool_x = np.array([1, 0, 0])
        tool_y = np.array([0, 0, -1])
        tool_z = np.cross(tool_x, tool_y)
        grasp_rotation = np.c_[tool_x, tool_y, tool_z]
        return grasp_translation, grasp_rotation

    def get_pose_information(self, meshes, grasp_type, width, vis=False):
        facenet_thumb = self.grasp_types[str(grasp_type+1)]['facenet_thumb']
        facenet_index = self.grasp_types[str(grasp_type+1)]['facenet_index']
        if len(facenet_thumb) != 0:
            translation, rotation = self.common_grasp_information(meshes, facenet_thumb, facenet_index, width, grasp_type)
        else:
            translation, rotation = self.special_grasp_information(meshes, facenet_index, width, grasp_type)
        if vis:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            grasp_matrix = np.vstack((np.hstack((rotation, translation.reshape((3, 1)))), np.array((0, 0, 0, 1))))
            frame_grasp = o3d.geometry.TriangleMesh.create_coordinate_frame(0.03).transform(grasp_matrix)
            g = Grasp().transform(grasp_matrix)
            g.width = width * 0.01
            sphere = o3d.geometry.TriangleMesh.create_sphere(0.002,20).translate(g.translation)
            o3d.visualization.draw_geometries([meshes, frame, frame_grasp, g.to_open3d_geometry(), sphere])
        return translation, rotation  

            
            
            






            
            
            

















if __name__ == '__main__':
    path_6d = '../InspireHandR_mesh/mesh/inspire_hand_routine_to_angle-use.xlsx'
    path_12d = '../InspireHandR_mesh/mesh/driver_routine_to_angle.xls'

    source_stl_path = '../InspireHandR_mesh/urdf-five3/meshes/'
    output_path = '../InspireHandR_mesh/mesh/'
    json_path = '../InspireHandR_mesh/mesh/width_12Dangle_6Dangle.json'

    # simplified_stl_path = '../InspireHandR_mesh/urdf-five3/meshes_simplified1/'
    # simplified_out_path = '../InspireHandR_mesh/mesh/simplified/'
    # simplified_json_path = '../InspireHandR_mesh/mesh/width_12Dangle_6Dangle_simplified.json'

    urdf_path = '../InspireHandR_mesh/urdf-five3/robots/urdf-five3.urdf'

    angles = read_excel_6Dangle_to_12Dangle(path_6d, path_12d)
    angles_to_stl(angles, source_stl_path, output_path, json_path, urdf_path, if_source=True, vis=False)
    # angles_to_stl(angles, simplified_stl_path, simplified_out_path, simplified_json_path, urdf_path, if_source=False, vis=False)





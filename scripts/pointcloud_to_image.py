import json
import os
from unittest.util import sorted_list_difference
from tqdm import tqdm
# import torch
import yaml
import cv2
import numpy as np
from PIL import Image
import open3d as o3d
import copy
import math

def read_picture(path):
    color_path = os.path.join(path, 'color.png')
    depth_pth = os.path.join(path, 'depth.png')
    color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    depth = np.array(Image.open(depth_pth))

    fx, fy = 912.898, 912.258
    cx, cy = 629.536, 351.637
    s = 1000.0

    xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depth / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0.45) & (points_z < 0.88) & (points_x > -0.3) & (points_x < 0.3) & (points_y > -0.2) & (points_y < 0.3)


    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = color[mask].astype(np.float32)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud
    
def read_json(data_path, mesh_path, voxel_size, DEBUG=True):
    '''
    **input:**
    - data_path: the path of collected data
    - mesh_path: the path of source meshes and pointclouds of meshes
    **output:**
    - InspireHandR_grasp: the InspireHandR pose of camera coordinate system, it will be used to grasp object
    - two_fingers_grasp: the two_fingers pose of camera coordinate system
    - InspireHandR_ggarray_proposals: the InspireHandR proposals' pose of camera coordinate system
    - two_fingers_ggarray_proposals: the two_fingers proposals' pose of camera coordinate system
    - base_2_tcp_ready: numpy array of shape (4,4) of the tcp pose while camera takes photo for object
    - tcp_2_camera: numpy array of shape (4,4) of the tcp to camera transformation matrix
    - source_mesh: the source mesh of InspireHandR, it will be used to grasp object
    - result: bool, if success
    '''
    json_path = os.path.join(data_path, 'information.json')
    with open(json_path, 'r') as f:
        information = json.load(f)
    two_fingers_grasp_json = information['two_fingers_pose']
    InspireHandR_grasp_json = information['InspiredHandR_pose']
    base_2_tcp1 = information['base_2_tcp1']
    base_2_tcp1_backup = information['base_2_tcp1_backup']
    tcp_2_gripper = information['tcp_2_gripper']
    base_2_TwoFingersGripper_pose = information['base_2_TwoFingersGripper_pose']
    tcp_2_camera = information['tcp_2_camera']
    base_2_tcp_ready = information['base_2_tcp_ready']
    two_fingers_ggarray_proposals = np.array(information['two_fingers_ggarray_proposals'])
    InspireHandR_ggarray_proposals = np.array(information['InspireHandR_ggarray_proposals'])
    two_fingers_ggarray_source_saved = np.array(information['two_fingers_ggarray_source'])
    InspireHandR_ggarray_source_saved = np.array(information['InspireHandR_ggarray_source_saved'])

    InspireHandR_grasp = InspireHandRGrasp(np.array(InspireHandR_grasp_json))
    two_fingers_grasp = Grasp(np.array(two_fingers_grasp_json))

    source_mesh = InspireHandR_grasp.load_mesh(mesh_path, two_fingers_grasp)
    source_mesh_pointclouds = InspireHandR_grasp.load_mesh_pointclouds(mesh_path, two_fingers_grasp, voxel_size=voxel_size)

    if DEBUG:
        two_fingers_2_InspireHandR_json_path = os.path.join(mesh_path, 'width_12Dangle_6Dangle.json')
        with open(two_fingers_2_InspireHandR_json_path, 'r') as f:
            two_fingers_2_InspireHandR_json = json.load(f)
            rotation = two_fingers_2_InspireHandR_json[InspireHandR_grasp.get_grasp_type_with_finger_name()][
                str(round(InspireHandR_grasp.width * 100, 1))]['rotation']
            translation = two_fingers_2_InspireHandR_json[InspireHandR_grasp.get_grasp_type_with_finger_name()][
                str(round(InspireHandR_grasp.width * 100, 1))]['translation']
            mesh_offset = np.vstack((np.hstack(
                (np.array(rotation), np.array(translation).reshape((3, 1)))),
                                     np.array((0, 0, 0, 1.0))))

        two_fingers_grasp_translation = two_fingers_grasp.translation
        two_fingers_grasp_rotation = two_fingers_grasp.rotation_matrix
        camera_TwoFingers_pose = np.vstack((np.hstack(
            (np.array(two_fingers_grasp_rotation), np.array(two_fingers_grasp_translation).reshape((3, 1)))),
                                            np.array((0, 0, 0, 1.0))))

        InspireHandR_grasp_translation = InspireHandR_grasp.translation
        InspireHandR_grasp_rotation = InspireHandR_grasp.rotation_matrix
        camera_InspireHandR_pose = np.vstack((np.hstack(
            (np.array(InspireHandR_grasp_rotation), np.array(InspireHandR_grasp_translation).reshape((3, 1)))),
                                              np.array((0, 0, 0, 1.0))))

        a = np.dot(np.dot(base_2_tcp1, tcp_2_gripper), mesh_offset)
        b = base_2_TwoFingersGripper_pose
        c = np.dot(np.dot(base_2_tcp_ready, tcp_2_camera), camera_InspireHandR_pose)
        d = np.dot(np.dot(base_2_tcp_ready, tcp_2_camera), camera_TwoFingers_pose)

    return InspireHandR_grasp, two_fingers_grasp, InspireHandR_ggarray_proposals, two_fingers_ggarray_proposals, \
           base_2_tcp_ready, tcp_2_camera, base_2_tcp1, source_mesh, source_mesh_pointclouds

def read_file(data_path, mesh_path, image_output_path, voxel_size=0.002):
    '''
        **input:**
        - data_path: the path of collected data
        - mesh_path: the path of source meshes and the pointclouds of meshes
        - DEBUG: test the related matrix and visualization
        **output:**
        - pointclouds: the pictures of pointclouds
        - InspireHandR_grasps: the InspireHandR poses of camera coordinate system, they will be used to grasp object
        - InspireHandR_ggarrays_proposals: the InspireHandR proposals' poses of camera coordinate system
        - two_fingers_grasps: the two_fingers pose of camera coordinate system, the object_id is 0
        - two_fingers_ggarrays_proposals: the two_fingers proposals' poses of camera coordinate system, the object_ids are 0
        - results_if_success: bool, if success
    '''
    i = 0
    sorted_data_path = sorted(os.listdir(data_path))
    for file in tqdm(sorted_data_path):
        i = i + 1
        data = os.path.join(data_path, file)
        pointcloud = read_picture(data)
        try:
            all_information = read_json(data, mesh_path, voxel_size)
            InspireHandR_grasp = all_information[0]
            two_fingers_grasp = all_information[1]
            InspireHandR_ggarray_proposals = all_information[2]
            two_fingers_ggarray_proposals = all_information[3]
            base_2_tcp_ready = all_information[4]
            tcp_2_camera = all_information[5]
            source_mesh = all_information[7]
            source_mesh_pointclouds = all_information[8]

            FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
            pcd = copy.deepcopy(pointcloud)
            pcd.transform(np.dot(base_2_tcp_ready, tcp_2_camera))
            source_mesh.transform(np.dot(base_2_tcp_ready, tcp_2_camera))
            source_mesh.compute_vertex_normals()
            source_mesh_pointclouds.transform(np.dot(base_2_tcp_ready, tcp_2_camera))
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=source_mesh_pointclouds, voxel_size=voxel_size)
            two_fingers = two_fingers_grasp.to_open3d_geometry()
            two_fingers.transform(np.dot(base_2_tcp_ready, tcp_2_camera))

            z_angle = 86.5
            x_angle = 45
            rotation_z = np.array([[math.cos(math.pi*(z_angle/180)), math.sin(math.pi*(z_angle/180)), 0, 0], 
                                    [-math.sin(math.pi*(z_angle/180)), math.cos(math.pi*(z_angle/180)), 0, 0], 
                                    [0, 0, 1, 0], 
                                    [0, 0, 0, 1]])
            rotation_x = np.array([[1, 0, 0, 0], 
                                    [0, math.cos(math.pi*(x_angle/180)), math.sin(math.pi*(x_angle/180)), 0], 
                                    [0, -math.sin(math.pi*(x_angle/180)), math.cos(math.pi*(x_angle/180)), 0], 
                                    [0, 0, 0, 1]])
            final_mat = np.dot(rotation_x, rotation_z)
            pcd = pcd.transform(final_mat)
            source_mesh = source_mesh.transform(final_mat)
            two_fingers = two_fingers.transform(final_mat)

            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=True) #works for me with False, on some systems needs to be true
            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            vis.add_geometry(source_mesh)
            vis.update_geometry(source_mesh)
        except:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=True) #works for me with False, on some systems needs to be true
        
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(image_output_path, file + '.png'))
        vis.destroy_window()

def convert_to_second(time):
    hour, minute, second = time.split('_')[1].split('.')[0].split('-')
    return int(hour) * 3600 + int(minute) * 60 + int(second)
    
def put_text(img, type):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if type == '_left':
        text = 'right view'
        img = cv2.resize(img[100:,254:1274], (1528, 836))
    if type == '_right':
        text = 'left view'
        img = cv2.resize(img[100:,254:1274], (1528, 836))
    if type == '_up':
        text = 'top view'
        img = cv2.resize(img[140:697,382:1146], (1528, 836))
    if type == '_forward':
        text = 'frontal view'
        img = cv2.resize(img[140:697,254:1274], (1528, 836))
    cv2.putText(img, text, (150, 130), font, 3, (0, 0, 255), 6)
    return img

def convert_images_to_video(data_path, video_output_path):
    fps = 1
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_output_path, fourcc, fps, (1528*2, 836*2))

    angle = ['_left', '_right', '_forward', '_up']
    sorted_data_path = sorted(os.listdir(data_path + '_left'))
    joint_image = np.zeros((836*2, 1528*2, 3), np.uint8)
    for id, direction in enumerate(angle):
        image_path = os.path.join(data_path + direction, sorted_data_path[0])
        img = cv2.imread(image_path)
        img = put_text(img, direction)
        print(img.shape)
        if direction == '_left':
            joint_image[0:836, 0:1528] = img
        if direction == '_right':
            joint_image[0:836, 1528:1528*2] = img
        if direction == '_forward':
            joint_image[836:836*2, 0:1528] = img
        if direction == '_up':
            joint_image[836:836*2, 1528:1528*2] = img

    for i in range(90):
        video.write(joint_image)

    start = convert_to_second(sorted_data_path[0])   
    for image in tqdm(sorted_data_path[1:]):
        num_repeat_image = convert_to_second(image) - start
        start = convert_to_second(image)
        joint_image = np.zeros((836*2, 1528*2, 3), np.uint8)
        for id, direction in enumerate(angle):
            image_path = os.path.join(data_path + direction, image)
            img = cv2.imread(image_path)
            img = put_text(img, direction)
            if direction == '_left':
                joint_image[0:836, 0:1528] = img
            if direction == '_right':
                joint_image[0:836, 1528*1:1528*2] = img
            if direction == '_forward':
                joint_image[836*1:836*2, 0:1528] = img
            if direction == '_up':
                joint_image[836*1:836*2, 1528*1:1528*2] = img

        for i in range(num_repeat_image):
            video.write(joint_image)  # 把图片写进视频

    joint_image = np.zeros((836*2, 1528*2, 3), np.uint8)
    for id, direction in enumerate(angle):
        image_path = os.path.join(data_path + direction, sorted_data_path[-1])
        img = cv2.imread(image_path)
        img = put_text(img, direction)
        if direction == '_left':
            joint_image[0:836, 0:1528] = img
        if direction == '_right':
            joint_image[0:836, 1528*1:1528*2] = img
        if direction == '_forward':
            joint_image[836*1:836*2, 0:1528] = img
        if direction == '_up':
            joint_image[836*1:836*2, 1528*1:1528*2] = img
    for i in range(4):
        video.write(joint_image)
    
    video.release()  # 释放

def video_speedup(in_path, out_path, source_items_path, grasp_time=11, speedup_rate=3):
    source_video = VideoFileClip(in_path)
    sorted_data_path = []
    with open(source_items_path, 'r') as f:
        for line in f.readlines():
            sorted_data_path.append(line.strip())
    
    start = [0, convert_to_second(sorted_data_path[0])]
    re_video = []
    sum_time = 0
    for image in tqdm(sorted_data_path[1:]):
        one_grasp_time = convert_to_second(image) - start[1]
        end_time = one_grasp_time + start[0]
      
        sub_video = source_video.subclip(start[0], end_time)

        wait_time = one_grasp_time - grasp_time
        
        wait_video = sub_video.subclip(0, wait_time)
        wait_video = wait_video.set_fps(source_video.fps * speedup_rate).fx(vfx.speedx, speedup_rate)
        
        txt_clip_wait = TextClip("10 x speed", fontsize = 95, color = 'red') 
        txt_clip_wait = txt_clip_wait.set_pos((0.68, 0.08), relative=True).set_duration(wait_time/speedup_rate) 

        wait_video = CompositeVideoClip([wait_video, txt_clip_wait])
                
        grasp_video = sub_video.subclip(wait_time, one_grasp_time)
        
        txt_clip_grasp = TextClip("1 x speed", fontsize = 95, color = 'red') 
        txt_clip_grasp = txt_clip_grasp.set_pos((0.68, 0.08), relative=True).set_duration(grasp_time)  

        grasp_video = CompositeVideoClip([grasp_video, txt_clip_grasp])
     
        re_video.append(wait_video)
        re_video.append(grasp_video)

        sum_time = sum_time + wait_time/speedup_rate + grasp_time
        start = [end_time, convert_to_second(image)]
    
    
    end_time = source_video.duration - start[0]
    end_video = source_video.subclip(start[0], source_video.duration).set_fps(source_video.fps * speedup_rate).fx(vfx.speedx, speedup_rate)
    txt_clip_end = TextClip("10 x speed", fontsize = 95, color = 'red') 
    txt_clip_end = txt_clip_end.set_pos((0.68, 0.08), relative=True).set_duration(end_time/speedup_rate) 
    txt_clip_net = TextClip("graspnet.net", fontsize = 95, color = 'black') 
    txt_clip_net = txt_clip_net.set_pos((0.34, 0.02), relative=True).set_duration(end_time/speedup_rate) 
    end_video = CompositeVideoClip([end_video, txt_clip_end, txt_clip_net])
    
    re_video.append(end_video)
    re_video = concatenate_videoclips(re_video)
    re_video = re_video.without_audio()

    
    txt_clip_net = TextClip("graspnet.net", fontsize = 95, color = 'black') 
    txt_clip_net = txt_clip_net.set_pos((0.34, 0.02), relative=True).set_duration(sum_time) 
    re_video = CompositeVideoClip([re_video, txt_clip_net])
    
    re_video.write_videofile(out_path, threads=4, audio = False, progress_bar = False)   

def video_speedup2(in_path, video_output_path, source_items_path, grasp_time=11, speedup_rate=3):
    font = cv2.FONT_HERSHEY_SIMPLEX
    video_cap = cv2.VideoCapture(in_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    save_video = cv2.VideoWriter(video_output_path, fourcc, fps, (1920, 1080))

    sorted_data_path = []
    with open(source_items_path, 'r') as f:
        for line in f.readlines():
            sorted_data_path.append(line.strip())
    
    start = [0, convert_to_second(sorted_data_path[0])]

    frame_count = 0
    all_frames = []
    
    for id, image in enumerate(tqdm(sorted_data_path[1:])):
        one_grasp_time = convert_to_second(image) - start[1]
        total_imges_num = one_grasp_time * fps

        frame_count = frame_count + total_imges_num

        wait_images_num = ((one_grasp_time - grasp_time) / one_grasp_time) * total_imges_num
        grasp_images_num = total_imges_num - wait_images_num
        for i in range(int(total_imges_num)): 
            ret, frame = video_cap.read()
            if i % speedup_rate == 0 and i < wait_images_num:
                cv2.putText(frame, "10 x speed", (1305, 186), font, 3, (0, 0, 255), 6)
                cv2.putText(frame, "graspnet.net", (653, 121), font, 3, (0, 0, 0), 6)
                save_video.write(frame)
            elif i >= wait_images_num:
                cv2.putText(frame, "10 x speed", (1305, 186), font, 3, (0, 0, 255), 6)
                cv2.putText(frame, "graspnet.net", (653, 121), font, 3, (0, 0, 0), 6)
                save_video.write(frame)
            
        end_time = one_grasp_time + start[0]
        start = [end_time, convert_to_second(image)]
        

    for i in range(int(video_cap.get(7) - frame_count)):
        ret, frame = video_cap.read()
        if not ret:
            break
        cv2.putText(frame, "10 x speed", (1305, 186), font, 3, (0, 0, 255), 6)
        cv2.putText(frame, "graspnet.net", (653, 121), font, 3, (0, 0, 0), 6)
        if i % speedup_rate == 0:
            save_video.write(frame)
        
    
    save_video.release()

def video_speedup3(in_path, video_output_path, speedup_rate=10):
    font = cv2.FONT_HERSHEY_SIMPLEX
    video_cap = cv2.VideoCapture(in_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)

    video_clip_path = video_output_path.replace('.mp4', '_clip.mp4')
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    save_video = cv2.VideoWriter(video_output_path, fourcc, fps, (1920, 1080))
    fourcc_clip = cv2.VideoWriter_fourcc(*"mp4v")
    save_video_clip = cv2.VideoWriter(video_clip_path, fourcc_clip, fps, (1920, 1080))
    ret = True
    id = 0
    while ret:
        ret, frame = video_cap.read()
        if id > fps * (75 * 60 + 10):
            break
        save_video_clip.write(frame)
        if id % speedup_rate == 0:
            cv2.putText(frame, "2 x speed", (1305, 186), font, 3, (0, 0, 255), 6)
            cv2.putText(frame, "graspnet.net", (653, 121), font, 3, (0, 0, 0), 6)
            save_video.write(frame)
        id = id + 1
    save_video.release()
    save_video_clip.release()

def generate_video_time(path, out_path):
    with open(out_path, 'w')as f:
        for item in sorted(os.listdir(path)):
            f.write(item + '\n')

def get_success_rate(path):
    grasp_type = [[0, 0, 0] for i in range(20)]
    for pose_type in os.listdir(path):
        pose_id = pose_type.split('pose')[1]
        pose_type_path = os.path.join(path, pose_type)
        for item in os.listdir(pose_type_path):
            information_path = os.path.join(pose_type_path, item, 'information.json')
            with open(information_path, 'r') as f:
                information = json.load(f)
            result = information['result']
            if result:
                grasp_type[int(pose_id)][0] += 1
            else:
                grasp_type[int(pose_id)][1] += 1
            grasp_type[int(pose_id)][2] += 1
    su = [0, 0, 0]
    for pose_id, pose_num in enumerate(grasp_type):
        success_rate = pose_num[0] / (pose_num[2]+0.00000001)
        su[0] += pose_num[0]
        su[1] += pose_num[1]
        su[2] += pose_num[2]
        print('pose type: {}, success rate: {}, success num: {}, fail num: {}, sum: {}'.format(pose_id, success_rate, pose_num[0], pose_num[1], pose_num[2]))
    print(su[0]/su[2])

def get_success_rate_path(path):
    grasp_type = [[0, 0, 0] for i in range(20)]
    for item in os.listdir(path):
        information_path = os.path.join(path, item, 'information.json')
        with open(information_path, 'r') as f:
            information = json.load(f)
        result = information['result']
        g_type = information['InspiredHandR_pose_finger_type']
        if result:
            grasp_type[int(g_type)][0] += 1
        else:
            grasp_type[int(g_type)][1] += 1
        grasp_type[int(g_type)][2] += 1
    su = [0, 0, 0]
    for pose_id, pose_num in enumerate(grasp_type):
        success_rate = pose_num[0] / (pose_num[2]+0.00000001)
        su[0] += pose_num[0]
        su[1] += pose_num[1]
        su[2] += pose_num[2]
        print('pose type: {}, success rate: {}, success num: {}, fail num: {}, sum: {}'.format(pose_id, success_rate, pose_num[0], pose_num[1], pose_num[2]))
    print('success rate: {}, success num: {}, fail num: {}, sum: {}'.format(su[0]/su[2], su[0], su[1], su[2]))

if __name__ == '__main__':
    # NOTE: These are hardcoded test paths from the original author's machine.
    # Uncomment and change to your own data directory if you need success-rate
    # statistics.
    # path = '/home/ubuntu/data/hengxu/data/dh3/dh3_test/obj40'
    # path = '/home/ubuntu/data/hengxu/data/inspire/inspire_test_single_point/obj40/test_model0_collision'
    # path = '/home/ubuntu/data/hengxu/data/inspire/inspire_test_single_point/obj140/test0'
    # get_success_rate_path(path)
    pass
    

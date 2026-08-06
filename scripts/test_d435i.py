import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# 配置 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()

# 启用彩色流和深度流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("正在启动 D435i 相机...")
pipeline.start(config)

print("相机启动成功！正在采集一帧点云（请将物体或手置于相机下方）...")
try:
    # 等待一帧数据
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        print("未能获取到有效的深度或彩色帧！")
    else:
        # 使用 Open3D 将其转换为点云并可视化
        # 1. 转换为 Open3D 的 RGBD 图像
        depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
        color_image = o3d.geometry.Image(np.asanyarray(color_frame.get_data()))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, depth_image, convert_rgb_to_intensity=False
        )
        
        # 2. 获取相机内参
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, 
            intrinsics.fx, intrinsics.fy, 
            intrinsics.ppx, intrinsics.ppy
        )
        
        # 3. 生成点云
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cam_intrinsic)
        
        # 4. 翻转点云以适应常规视角
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        print(f"成功生成点云！点云包含 {len(pcd.points)} 个点。")
        print("正在弹窗显示点云视窗（关闭视窗后程序结束）...")
        
        # 5. 可视化
        o3d.visualization.draw_geometries([pcd], window_name="D435i PointCloud Test")

finally:
    pipeline.stop()
    print("相机已安全关闭。")

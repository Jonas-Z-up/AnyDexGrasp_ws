
import pyrealsense2 as rs
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)
profile = cfg.get_stream(rs.stream.color)
intr = profile.as_video_stream_profile().get_intrinsics()
print(intr)  # 获取内参 width: 640, height: 480, ppx: 319.115, ppy: 234.382, fx: 597.267, fy: 597.267, model: Brown Conrady, coeffs: [0, 0, 0, 0, 0] 319.1151428222656
print(intr.ppx)  # 获取指定某个内参

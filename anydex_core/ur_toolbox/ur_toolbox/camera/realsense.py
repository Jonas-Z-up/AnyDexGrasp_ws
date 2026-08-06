import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import json
import os

class RealSense():
    def __init__(self, serial = '035622060973', frame_rate = 30, resolution = (1280,720), timeout_ms=15000, warmup_frames=5):
        '''
        **Input:**

        - frame_rate: int of how many frames to take in one second. Don't exceed the maximum value for each resolution.

        - resolution: tuple of ints, (width, height).
        '''
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.serial = serial
        self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, frame_rate)
        self.config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, frame_rate)
        self.pipeline.start(self.config)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.colorizer = rs.colorizer()

        frames = None
        try:
            for _ in range(warmup_frames):
                frames = self.pipeline.wait_for_frames(timeout_ms)
        except RuntimeError as exc:
            self.pipeline.stop()
            raise RuntimeError(
                f"RealSense {self.serial} started but did not deliver frames within {timeout_ms} ms. "
                "If RGB works but depth times out, replug the camera, change USB3 port/cable, "
                "or reset the device with a RealSense tool before running AnyDexGrasp."
            ) from exc

        aligned_frames = self.align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            self.pipeline.stop()
            raise RuntimeError("RealSense started, but aligned depth/color frames are incomplete.")

        color_profile = color_frame.get_profile()
        cvsprofile = rs.video_stream_profile(color_profile)
        color_intrin = cvsprofile.get_intrinsics()
        print(color_intrin)
        color_intrin_part = [color_intrin.ppx, color_intrin.ppy, color_intrin.fx, color_intrin.fy]
        print(color_intrin_part)
        os.makedirs('logs/preview', exist_ok=True)
        with open('logs/preview/realsense_intrinsics.json', 'w') as f:
            json.dump({
                'width': color_intrin.width,
                'height': color_intrin.height,
                'ppx': color_intrin.ppx,
                'ppy': color_intrin.ppy,
                'fx': color_intrin.fx,
                'fy': color_intrin.fy,
            }, f, indent=2)

    def advanced():
        import json

        DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A", "0B5C"]

        
        ctx = rs.context()
        ds5_dev = rs.device()
        dev = ctx.query_devices()[0]
        try:
            advnc_mode = rs.rs400_advanced_mode(dev)
            print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

            # Loop until we successfully enable advanced mode
            while not advnc_mode.is_enabled():
                print("Trying to enable advanced mode...")
                advnc_mode.toggle_advanced_mode(True)
                # At this point the device will disconnect and re-connect.
                print("Sleeping for 5 seconds...")
                time.sleep(5)
                # The 'dev' object will become invalid and we need to initialize it again
                dev = ctx.query_devices()[0]
                advnc_mode = rs.rs400_advanced_mode(dev)
                print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

            # Get each control's current value
            print("Depth Control: \n", advnc_mode.get_depth_control())
            print("RSM: \n", advnc_mode.get_rsm())
            print("RAU Support Vector Control: \n", advnc_mode.get_rau_support_vector_control())
            print("Color Control: \n", advnc_mode.get_color_control())
            print("RAU Thresholds Control: \n", advnc_mode.get_rau_thresholds_control())
            print("SLO Color Thresholds Control: \n", advnc_mode.get_slo_color_thresholds_control())
            print("SLO Penalty Control: \n", advnc_mode.get_slo_penalty_control())
            print("HDAD: \n", advnc_mode.get_hdad())
            print("Color Correction: \n", advnc_mode.get_color_correction())
            print("Depth Table: \n", advnc_mode.get_depth_table())
            print("Auto Exposure Control: \n", advnc_mode.get_ae_control())
            print("Census: \n", advnc_mode.get_census())

            as_json_object = json.loads(serialized_string)

            # We can also load controls from a json string
            # For Python 2, the values in 'as_json_object' dict need to be converted from unicode object to utf-8
            if type(next(iter(as_json_object))) != str:
                as_json_object = {k.encode('utf-8'): v.encode("utf-8") for k, v in as_json_object.items()}
            # The C++ JSON parser requires double-quotes for the json object so we need
            # to replace the single quote of the pythonic json to double-quotes
            json_string = str(as_json_object).replace("'", '\"')
            advnc_mode.load_json(json_string)

        except Exception as e:
            print(e)
            pass

    def get_rgbd_image(self, return_pcd=False, return_color_depth=False):
        '''
        **Input:**

        - return_pcd: bool of whether to return open3d.Pointcloud.

        - return_color_depth: bool of whether to return color_depth_image.

        **Output:**

        - tuple of color_image, depth_image.

        - color_image: np.array of shape (height, width, 3).

        - depth_image: np.array of shape (height, width), dtype = np.uint16.
        '''
        frames = self.pipeline.wait_for_frames()
        # frames.keep()
        frameset = self.align.process(frames)
        depth_frame = frameset.get_depth_frame()
        color_frame = frameset.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        colorized_depth = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())


        if return_pcd:
            colors = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) / 255.0
            depths = depth_image
            param = o3d.camera.PinholeCameraParameters()
            param.intrinsic.set_intrinsics(1280,720, 908.435, 908.679, 650.366, 367.277)
            intrinsics = param.intrinsic.intrinsic_matrix
            fx, fy = intrinsics[0,0], intrinsics[1,1]
            cx, cy = intrinsics[0,2], intrinsics[1,2]
            s = 1000.0
            
            xmap, ymap = np.arange(colors.shape[1]), np.arange(colors.shape[0])
            xmap, ymap = np.meshgrid(xmap, ymap)

            points_z = depths / s
            points_x = (xmap - cx) / fx * points_z
            points_y = (ymap - cy) / fy * points_z

            mask = (points_z > 0)
            points = np.stack([points_x, points_y, points_z], axis=-1)
            points = points[mask]
            colors = colors[mask]
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)
        if not (return_pcd or return_color_depth):
            return color_image, depth_image
        if (not return_color_depth) and return_pcd:
            return color_image, depth_image, cloud
        if (not return_pcd) and return_color_depth:
            return color_image, depth_image, colorized_depth
        if return_color_depth and return_pcd:
            return color_image, depth_image, cloud, colorized_depth

            

    def get_rgb_image(self):
        '''
        **Output:**

        - color_image: np.array of shape (height, width, 3).
        '''
        self.frames = self.pipeline.wait_for_frames()
        color_frame = self.frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def get_depth_image(self):
        '''
        **Output:**

        - depth_image: np.array of shape (height, width), dtype = np.uint16.
        '''
        self.frames = self.pipeline.wait_for_frames()
        depth_frame = self.frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        return depth_image

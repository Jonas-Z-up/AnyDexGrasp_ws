import os, sys
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.fonts=false"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'anydex_core'))

import pdb

from ur_toolbox.camera import RealSense
from multiprocessing import shared_memory
import numpy as np
import cv2
import os
import time
from camera_snapshot import META_NAME, META_SIZE

DEBUG = True
save_path = "capture_b06"

def create_shared_memory(name, size):
    try:
        return shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        old_shm = shared_memory.SharedMemory(name=name)
        old_shm.close()
        old_shm.unlink()
        return shared_memory.SharedMemory(name=name, create=True, size=size)

# function to display the coordinates of
# of the points clicked on the image 
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks

    fx, fy = 910.673, 908.948
    cx, cy = 655.339, 371.053
    s = 1000.0

    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    points = np.stack([points_x, points_y, points_z], axis=-1)
    # displaying the coordinates
    # on the image window
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(colors, str(int(points[y][x][0]*1000)) + ',' +
                str(int(points[y][x][1]*1000)) + ',' +
                str(int(points[y][x][2]*1000)), (x,y), font,
                1, (255, 0, 0), 2)
    cv2.imshow('image', colors)




camera = RealSense(serial = '147122074820', frame_rate = 30)
if DEBUG:
    for i in range(10):
        colors, depths = camera.get_rgbd_image()
else:
    for i in range(10):
        depths = camera.get_depth_image()
        

shm_depth = create_shared_memory('realsense_depth', depths.nbytes)
depthbuf = np.ndarray(depths.shape, dtype=depths.dtype, buffer=shm_depth.buf)
shm_meta = create_shared_memory(META_NAME, META_SIZE)
metadata = np.ndarray((2,), dtype=np.int64, buffer=shm_meta.buf)
metadata[:] = 0

if DEBUG:
    colors = (cv2.cvtColor(colors, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
    shm_color = create_shared_memory('realsense_color', colors.nbytes)
    colorbuf = np.ndarray(colors.shape, dtype=colors.dtype, buffer=shm_color.buf)

try:
    cnt = 0
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)

    while True:
        colors = None
        
        if DEBUG:
            colors, depths = camera.get_rgbd_image()
            captured_ns = time.time_ns()
            cv2.imshow("RealSense D435i", colors)
            cv2.waitKey(1)
            colors = (cv2.cvtColor(colors, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
            # cv2.imwrite(os.path.join(save_path, 'color_%03d.png'%cnt), colors)
            # np.save(os.path.join(save_path, 'depth_%03d.npy'%cnt), depths)
            # print(depths.dtype)

            cnt += 1
            
            metadata[0] += 1
            colorbuf[:] = colors[:]
            depthbuf[:] = depths[:]
            metadata[1] = captured_ns
            metadata[0] += 1
        else:
            print(shm_depth.name)
            depths = camera.get_depth_image()
            metadata[0] += 1
            depthbuf[:] = depths[:]
            metadata[1] = time.time_ns()
            metadata[0] += 1
except KeyboardInterrupt:
    pass
finally:
    for shm in [shm_depth, shm_meta] + ([shm_color] if DEBUG else []):
        shm.close()
        shm.unlink()

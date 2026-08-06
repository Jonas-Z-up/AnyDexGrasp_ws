# example of displaying images from realsense camera.

from ur_toolbox.camera import RealSense
import cv2

r = RealSense(frame_rate=6)
i = 0
while True:
    print(i)
    i += 1
    color_image  = r.get_rgb_image()
    print(color_image.shape)
    cv2.imshow('image', color_image)
    cv2.waitKey(2)
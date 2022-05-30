
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np
import mapper
import time
from PIL import Image

from ssd import SSD

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
ssd = SSD()

while True:
    if kinect.has_new_depth_frame():
        color_frame = kinect.get_last_color_frame()
        colorImage = color_frame.reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4)).astype(np.uint8)
        colorImage = cv2.flip(colorImage, 1)


        t1 = time.time()
        frame = kinect.get_last_color_frame()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        image, informations = ssd.detect_image(frame)
        frame = np.array(image)
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("video",frame)
        # cv2.imshow('depth',color_data[3])
        cv2.waitKey(1)



        # cv2.imshow('Test Color View', cv2.resize(colorImage, (int(1920 / 2.5), int(1080 / 2.5))))
        # depth_frame = kinect.get_last_depth_frame()
        # depth_img = depth_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width)).astype(np.uint8)
        # depth_img = cv2.flip(depth_img, 1)
        # cv2.imshow('Test Depth View', depth_img)
        # print(color_point_2_depth_point(kinect, _DepthSpacePoint, kinect._depth_frame_data, [100, 100]))
        # print(depth_points_2_world_points(kinect, _DepthSpacePoint, [[100, 150], [200, 250]]))
        # print(intrinsics(kinect).FocalLengthX, intrinsics(kinect).FocalLengthY, intrinsics(kinect).PrincipalPointX, intrinsics(kinect).PrincipalPointY)
        # print(intrinsics(kinect).RadialDistortionFourthOrder, intrinsics(kinect).RadialDistortionSecondOrder, intrinsics(kinect).RadialDistortionSixthOrder)
        # print(world_point_2_depth(kinect, _CameraSpacePoint, [0.250, 0.325, 1]))
        # img = depth_2_color_space(kinect, _DepthSpacePoint, kinect._depth_frame_data, show=False, return_aligned_image=True)
        depth_2_color_space(kinect, _DepthSpacePoint, kinect._depth_frame_data, show=True)
        # img = color_2_depth_space(kinect, _ColorSpacePoint, kinect._depth_frame_data, show=True, return_aligned_image=True)

    # Quit using q
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
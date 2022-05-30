#-------------------------------------#
#   调用摄像头或者视频进行检测
#   调用摄像头直接运行即可
#   调用视频可以将cv2.VideoCapture()指定路径
#   视频的保存并不难，可以百度一下看看
#-------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image
from ssd import SSD

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np
import ctypes
import math
import cv2 as cv
import time
import copy
import mapper

class Kinect(object):
    def __init__(self):
        self._kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
        self.depth_ori = None
        self.infrared_frame = None
        self.color_frame = None
        self.w_color = 1920#1920
        self.h_color = 1080#1080
        self.w_depth = 512#512
        self.h_depth = 424#424
        """————————————————(2019/5/10)——————————————————"""
        self.csp_type = _ColorSpacePoint * np.int(1920 * 1080)
        self.csp = ctypes.cast(self.csp_type(), ctypes.POINTER(_DepthSpacePoint))
        """————————————————(2019/9/4)——————————————————"""
        self.color = None
        self.depth = None
        self.depth_draw = None
        self.color_draw = None
        self.infrared = None
        self.first_time = True
    """————————————————(2019/5/1)——————————————————"""
    """获取最新的图像数据"""
    def get_the_last_color(self):
        """
        Time :2019/5/1
        FunC:获取最新的图像数据
        Input:无
        Return:无
        """
        if self._kinect.has_new_color_frame():
            # 获得的图像数据是二维的，需要转换为需要的格式
            frame = self._kinect.get_last_color_frame()
            # 返回的是4通道，还有一通道是没有注册的
            gbra = frame.reshape([self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4])
            # 取出彩色图像数据
            self.color_frame = gbra[:, :, 0:3]
            return self.color_frame
    """————————————————(2019/5/1)——————————————————"""
    """获取最新的深度数据"""
    def get_the_last_depth(self):
        """
        Time :2019/5/1
        FunC:获取最新的图像数据
        Input:无
        Return:无
        """
        if self._kinect.has_new_depth_frame():
            # 获得深度图数据
            frame = self._kinect.get_last_depth_frame()
            # 转换为图像排列
            image_depth_all = frame.reshape([self._kinect.depth_frame_desc.Height,
                                             self._kinect.depth_frame_desc.Width])
            self.depth_ori = image_depth_all[:,::-1]

            return self.depth_ori

    """————————————————(2019/5/1)——————————————————"""
    """将深度像素点匹配到彩色图像中"""
    def map_depth_point_to_color_point(self, depth_point):
        """
        Time :2019/5/1
        FunC: 将深度图像坐标映射到彩色坐标中，
        Input: depth_points:深度像素点，列表、数组格式,图像坐标，且为人眼视角
        Return: color_points:对应的彩色像素点，列表格式
        均采用图像坐标的形式
        """
        depth_point_to_color  = copy.deepcopy(depth_point)
        n = 0
        while 1:
            self.get_the_last_depth()
            self.get_the_last_color()
            if self.depth_ori is None:
                continue
            color_point = self._kinect._mapper.MapDepthPointToColorSpace(
                _DepthSpacePoint(511-depth_point_to_color[1], depth_point_to_color[0]), self.depth_ori[depth_point_to_color[0], 511-depth_point_to_color[1]])
            """————————————————(2019/6/22)——————————————————"""
            # cobot 第二次培训更改
            # color_point = self._kinect._mapper.MapDepthPointToColorSpace(
            #     _DepthSpacePoint(depth_point[0], depth_point[1]), self.depth[depth_point[1], depth_point[0]])
            """————————————————(2019/6/22)——————————————————"""
            if math.isinf(float(color_point.y)):
                n += 1
                if n >= 50000:
                    print('深度映射彩色，无效点')
                    color_point = [0, 0]
                    break
            else:
                color_point = [np.int0(color_point.y), 1920-np.int0(color_point.x)]  # 图像坐标，人眼视角
                break
        return color_point
    """————————————————(2019/5/10)——————————————————"""
    """将彩色像素数组映射到深度图像中"""
    def map_color_points_to_depth_points(self, color_points):
        """
        Time :2019/5/1
        FunC: 将深度图像坐标映射到彩色坐标中，输入的为图像坐标
        Input: depth_points:彩色像素点
        Return: color_points:对应的深度像素点，图像坐标
        """
        self.get_the_last_depth()
        self.get_the_last_color()
        self._kinect._mapper.MapColorFrameToDepthSpace(
            ctypes.c_uint(512 * 424), self._kinect._depth_frame_data, ctypes.c_uint(1920 * 1080), self.csp)
        depth_points = [self.map_color_point_to_depth_point(x, True) for x in color_points]
        return depth_points
    """————————————————(2019/5/1)——————————————————"""
    """将彩色像素点映射到深度图像中"""
    def map_color_point_to_depth_point(self, color_point, if_call_flg=False):
        """
        Time :2019/5/1
        FunC: 将深度图像坐标映射到彩色坐标中，输入的为图像坐标
        Input: depth_points:彩色像素点
        Return: color_points:对应的深度像素点，图像坐标
        """
        n = 0
        color_point_to_depth = copy.deepcopy(color_point)
        color_point_to_depth[1] = 1920 - color_point_to_depth[1]
        while 1:
            self.get_the_last_depth()
            self.get_the_last_color()
            # self.depth = cv.medianBlur(image_depth_all, 5)
            # print(self._kinect._depth_frame_data)
            if not if_call_flg:
                self._kinect._mapper.MapColorFrameToDepthSpace(
                    ctypes.c_uint(512 * 424), self._kinect._depth_frame_data, ctypes.c_uint(1920 * 1080), self.csp)
            if math.isinf(float(self.csp[color_point_to_depth[0]*1920+color_point_to_depth[1]-1].y)) or np.isnan(self.csp[color_point_to_depth[0]*1920+color_point_to_depth[1]-1].y):
                n += 1
                if n >= 50:
                    print('彩色映射深度，无效的点')
                    depth_point = [0, 0]
                    break
            else:
                self.cor = self.csp[color_point_to_depth[0]*1920+color_point_to_depth[1]-1].y
                try:
                    depth_point = [np.int0(self.csp[color_point_to_depth[0]*1920+color_point_to_depth[1]-1].y),
                                   np.int0(self.csp[color_point_to_depth[0]*1920+color_point_to_depth[1]-1].x)]
                except OverflowError as e:
                    print('彩色映射深度，无效的点')
                    depth_point = [0, 0]
                break
        depth_point[1] = 512-depth_point[1]
        return depth_point

        # depth_points = [self._kinect._mapper.MapColorPointToDepthSpace(_ColorSpacePoint(color_point[0],color_point[1]),self.color_frame[depth_point]))
        #                 for depth_point in depth_points]
        # return color_points
    """————————————————(2019/4/26)——————————————————"""
    """获得最新的彩色和深度图像以及红外图像"""
    def get_the_data_of_color_depth_infrared_image(self, Infrared_threshold = 16000):
        """
        # ————————查看是否有新的一帧————————
        :return:
        """
        # 访问新的RGB帧
        time_s = time.time()
        # print(self.first_time)
        if self.first_time:
            while 1:
                # print(1111)
                n = 0
                if self._kinect.has_new_color_frame():
                    print("color data!!!!!!!!")
                    #                 # 获得的图像数据是二维的，需要转换为需要的格式
                    frame = self._kinect.get_last_color_frame()
                    # 返回的是4通道，还有一通道是没有注册的
                    gbra = frame.reshape([self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4])
                    # 取出彩色图像数据
                    # self.color = gbra[:, :, 0:3]
                    self.color = gbra[:, :, 0:3][:,::-1,:]
                    # 这是因为在python中直接复制该图像的效率不如直接再从C++中获取一帧来的快
                    frame = self._kinect.get_last_color_frame()
                    # 返回的是4通道，还有一通道是没有注册的
                    gbra = frame.reshape([self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4])
                    # 取出彩色图像数据
                    # self.color_draw = gbra[:, :, 0:3][:,::-1,:]
                    self.color_draw = gbra[:, :, 0:3][:,::-1,:]
                    n += 1
                # 访问新的Depth帧
                if self._kinect.has_new_depth_frame():
                    print("depth data111!!!!!!!!!")
                    # 获得深度图数据
                    frame = self._kinect.get_last_depth_frame()
                    # 转换为图像排列
                    image_depth_all = frame.reshape([self._kinect.depth_frame_desc.Height,
                                                     self._kinect.depth_frame_desc.Width])
                    # 转换为（n，m，1） 形式
                    image_depth_all = image_depth_all.reshape(
                        [self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width, 1])
                    self.depth_ori = np.squeeze(image_depth_all)
                    self.depth = np.squeeze(image_depth_all)[:,::-1]

                    """————————————————(2019/5/11)——————————————————"""
                    # 获得深度图数据
                    frame = self._kinect.get_last_depth_frame()
                    # 转换为图像排列
                    depth_all_draw = frame.reshape([self._kinect.depth_frame_desc.Height,
                                                     self._kinect.depth_frame_desc.Width])
                    # 转换为（n，m，1） 形式
                    depth_all_draw = depth_all_draw.reshape(
                        [self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width, 1])
                    depth_all_draw[depth_all_draw >= 1500] = 0
                    depth_all_draw[depth_all_draw <= 500] = 0
                    depth_all_draw = np.uint8(depth_all_draw / 1501 * 255)
                    self.depth_draw = depth_all_draw[:,::-1,:]
                    n += 1
                t = time.time() - time_s
                print(n)
                time.sleep(2)
                if n == 2:
                    self.first_time = False
                    break
                elif t > 5:
                    print('未获取图像数据，请检查Kinect2连接是否正常')
                    break

        else:
            if self._kinect.has_new_color_frame():
                #                 # 获得的图像数据是二维的，需要转换为需要的格式
                frame = self._kinect.get_last_color_frame()
                # 返回的是4通道，还有一通道是没有注册的
                gbra = frame.reshape([self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4])
                # 取出彩色图像数据
                # self.color = gbra[:, :, 0:3]
                self.color = gbra[:, :, 0:3][:, ::-1, :]
                # 这是因为在python中直接复制该图像的效率不如直接再从C++中获取一帧来的快
                frame = self._kinect.get_last_color_frame()
                # 返回的是4通道，还有一通道是没有注册的
                gbra = frame.reshape([self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4])
                # 取出彩色图像数据
                # self.color_draw = gbra[:, :, 0:3][:,::-1,:]
                self.color_draw = gbra[:, :, 0:3][:, ::-1, :]

            # 访问新的Depth帧
            if self._kinect.has_new_depth_frame():
                # print("depth data2!!!!!")
                # 获得深度图数据
                frame = self._kinect.get_last_depth_frame()
                # 转换为图像排列
                image_depth_all = frame.reshape([self._kinect.depth_frame_desc.Height,
                                                 self._kinect.depth_frame_desc.Width])
                # 转换为（n，m，1） 形式
                image_depth_all = image_depth_all.reshape(
                    [self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width, 1])
                self.depth_ori = np.squeeze(image_depth_all)
                self.depth = np.squeeze(image_depth_all)[:, ::-1]

                """————————————————(2019/5/11)——————————————————"""
                # 获得深度图数据
                frame = self._kinect.get_last_depth_frame()
                # 转换为图像排列
                depth_all_draw = frame.reshape([self._kinect.depth_frame_desc.Height,
                                                self._kinect.depth_frame_desc.Width])
                # 转换为（n，m，1） 形式
                depth_all_draw = depth_all_draw.reshape(
                    [self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width, 1])
                depth_all_draw[depth_all_draw >= 1500] = 0
                depth_all_draw[depth_all_draw <= 500] = 0
                depth_all_draw = np.uint8(depth_all_draw / 1501 * 255)
                self.depth_draw = depth_all_draw[:, ::-1, :]

        return self.color, self.color_draw, self.depth, self.depth_draw
    """————————————————(2019/9/3)——————————————————"""
    """显示各种图像的视频流"""
    def Kinect_imshow(self,type_im='rgb'):
        """
        Time :2019/9/3
        FunC:
        Input: color_data
        Return: color_data
        """
        if type_im =='all':
            pass
        elif type_im =='rgb':
            pass
        elif type_im =='depth':
            pass
        elif type_im =='grared':
            pass

# Map Color Space to Depth Space (Image)
def color_2_depth_space(kinect, color_space_point, depth_frame_data, show=False, return_aligned_image=False):
    """
    :param kinect: kinect class
    :param color_space_point: _ColorSpacePoint from PyKinectV2
    :param depth_frame_data: kinect._depth_frame_data
    :param show: shows aligned image with color and depth
    :return: mapped depth to color frame
    """
    # import numpy as np
    # import ctypes
    # import cv2
    # Map Depth to Color Space
    depth2color_points_type = color_space_point * np.int(512 * 424)
    # print(kinect.csp_type)
    # print(depth2color_points_type)
    depth2color_points = ctypes.cast(depth2color_points_type(), ctypes.POINTER(color_space_point))
    kinect._mapper.MapDepthFrameToColorSpace(
        ctypes.c_uint(512 * 424), depth_frame_data, kinect._depth_frame_data_capacity, depth2color_points)
    # depth_x = depth2color_points[color_point[0] * 1920 + color_point[0] - 1].x
    # depth_y = depth2color_points[color_point[0] * 1920 + color_point[0] - 1].y
    colorXYs = np.copy(np.ctypeslib.as_array(depth2color_points, shape=(kinect.depth_frame_desc.Height * kinect.depth_frame_desc.Width,)))  # Convert ctype pointer to array
    colorXYs = colorXYs.view(np.float32).reshape(colorXYs.shape + (-1,))  # Convert struct array to regular numpy array https://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
    colorXYs += 0.5
    colorXYs = colorXYs.reshape(kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width, 2).astype(np.int)
    colorXs = np.clip(colorXYs[:, :, 0], 0, kinect.color_frame_desc.Width - 1)
    colorYs = np.clip(colorXYs[:, :, 1], 0, kinect.color_frame_desc.Height - 1)
    color_frame = kinect.get_last_color_frame()
    color_img = color_frame.reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4)).astype(np.uint8)
    align_color_img = np.zeros((424, 512, 4), dtype=np.uint8)
    align_color_img[:, :] = color_img[colorYs, colorXs, :]
    if show:
        cv2.imshow('img', cv2.flip(align_color_img, 1))
        cv2.waitKey(3000)
    if return_aligned_image:
        return align_color_img
    return colorXs, colorYs



# Map Color Points to Depth Points
def color_point_2_depth_point(kinect, depth_space_point, depth_frame_data, color_point):
    """

    :param kinect: kinect class
    :param depth_space_point: _DepthSpacePoint from PyKinectV2
    :param depth_frame_data: kinect._depth_frame_data
    :param color_point: color_point pixel location as [x, y]
    :return: depth point of color point
    """
    # Import here to optimize
    import numpy as np
    import ctypes
    # Map Color to Depth Space
    # Make sure that the kinect was able to obtain at least one color and depth frame, else the dept_x and depth_y values will go to infinity
    color2depth_points_type = depth_space_point * np.int(1920 * 1080)
    color2depth_points = ctypes.cast(color2depth_points_type(), ctypes.POINTER(depth_space_point))
    kinect._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512 * 424), depth_frame_data, ctypes.c_uint(1920 * 1080), color2depth_points)
    # Where color_point = [xcolor, ycolor]
    depth_x = color2depth_points[color_point[1] * 1920 + color_point[0] - 1].x
    depth_y = color2depth_points[color_point[1] * 1920 + color_point[0] - 1].y
    return [int(depth_x), int(depth_y)]

# Map a depth point to world point
def depth_point_2_world_point(kinect, depth_space_point, depthPoint):
    """
    :param kinect: kinect class
    :param depth_space_point: _DepthSpacePoint from PyKinectV2
    :param depthPoint: depth point as array [x, y]
    :return: return the camera space point
    """
    # Import here for optimization
    import numpy as np
    import ctypes
    depth_point_data_type = depth_space_point * np.int(1)
    depth_point = ctypes.cast(depth_point_data_type(), ctypes.POINTER(depth_space_point))
    depth_point.contents.x = depthPoint[0]
    depth_point.contents.y = depthPoint[1]
    world_point = kinect._mapper.MapDepthPointToCameraSpace(depth_point.contents, ctypes.c_ushort(512*424))
    return [world_point.x, world_point.y, world_point.z]  # meters


def depth2mi(depthValue):
    return depthValue * 0.001

def depth2xyz(u, v, depthValue):
    fx = 258.9839
    fy = 266.8131
    cx = 357.9089
    cy = 215.6056
    depth = depth2mi(depthValue)
    z = float(depth)
    x = float((u - cx) * z) / fx
    y = float((v - cy) * z) / fy
    result = [x, y, z]
    return result

ssd = SSD()
#-------------------------------------#
#   调用摄像头
#   capture=cv2.VideoCapture("1.mp4")
#-------------------------------------#
# capture=cv2.VideoCapture(0)
fps = 0.0

a = Kinect()
while(True):
    t1 = time.time()
    # 读取某一帧
    # ref,frame=capture.read()
    color_data = a.get_the_data_of_color_depth_infrared_image()
    frame = color_data[0]

    # print(frame)
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    # 进行检测
    image, informations = ssd.detect_image(frame)
    frame = np.array(image)
    temp = []
    # if informations != temp:
    #     print("realdata:\n "+informations)
    #     print("Not Null!")
    #     i = 0
    #     for information in informations:
    #         print(i + information)#label y,x(左上角) h w
    #         center_x=information[2]+information[4]/2
    #         center_y=information[1]+information[3]/2
    #         depthpoint = a.map_color_point_to_depth_point((center_x,center_y))
    #         print("depthpoint:"+ depthpoint)
    #         i+=1
    #     time.sleep(5)
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.namedWindow("video", 0)
    # cv2.resizeWindow("video", 960, 540)
    # cv2.imshow("video",frame)
    # cv2.waitKey(1)
    # cv2.imshow('depth',color_data[2])
    # cv2.waitKey(1)
    if informations != temp:
        # print("realdata:")
        # print(informations)
        # print("Not Null!")
        # time.sleep(2)
        i = 0
        for information in informations:
            # print(i)
            print(information)#label y,x(左上角) h w
            time.sleep(1)
            center_x=int(information[2]+information[4]/2)
            center_y=int(information[1]+information[3]/2)
            print(center_x,center_y)

            direction = [[1, 0], [0, 1], [1, -1], [-1, 1], [-1, 0], [0, -1], [1, 1], [-1, -1]]

            num = 0;
            while (True):
                x,y = mapper.color_point_2_depth_point(a._kinect,_DepthSpacePoint,a._kinect._depth_frame_data,[int(center_x),int(center_y)])
                # print(x, y)
                if x == 0 and y == 0:
                    center_x += direction[num][0]
                    center_y += direction[num][1]
                    continue

                depthpoint = depth_point_2_world_point(a._kinect, _DepthSpacePoint, [x,y])
                print("depthpoint:")
                print(depthpoint)

                if (depthpoint[2] > 20 and depthpoint[2] < 21):
                    # print("not qualified")
                    continue
                else:
                    break

            # depthpoint = a.map_color_point_to_depth_point([int(center_x),int(center_y)])
            # x,y,z = depth2xyz(depthpoint[0],depthpoint[1], color_data[2][depthpoint[0]][depthpoint[1]])
            # print("depthpoint:")
            # print(depthpoint)
            # print(x,y,z)

            i+=1
            print("break!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            time.sleep(10)
        # time.sleep(5)

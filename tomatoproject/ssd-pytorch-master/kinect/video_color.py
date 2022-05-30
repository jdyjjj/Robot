from __future__ import division
import cv2
#to show the image
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin
import time
import numpy as np
from PIL import Image

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np
import ctypes
import math
# import cv2 as cv
import time
import copy

green = (0, 255, 0)
red = (255, 0, 0)

def show(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, interpolation='nearest')

def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img

def find_biggest_contour(image):
    image = image.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask

def rectangle_coutour_red(image,contour):
    image_with_rectangle = image.copy()
    # easy function
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_with_rectangle, (x, y), (x + w, y + h), (0, 0, 255), 2, 8, 0)
    # add it
    # cv2.show(image_with_rectangle)
    return image_with_rectangle,x,y,w,h

def circle_contour_red(image, contour):
    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(contour)
    print(ellipse)
    cv2.ellipse(image_with_ellipse, ellipse, red, 2,cv2.LINE_AA)
    return image_with_ellipse


def find_strawberry_red(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    max_dimension = max(image.shape)
    scale = 700/max_dimension
    image = cv2.resize(image, None, fx=scale, fy=scale)
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])
    mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)
    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)
    mask = mask1 + mask2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    big_strawberry_contour, mask_strawberries = find_biggest_contour(mask_clean)
    overlay = overlay_mask(mask_clean, image)
    # circled = circle_contour_red(overlay, big_strawberry_contour)
    circled,x,y,w,h = rectangle_coutour_red(overlay, big_strawberry_contour)
    show(circled)

    #we're done, convert back to original color scheme
    bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)

    return bgr,x,y,w,h


#read the image


# image = cv2.imread("C:/Users/jdy/Desktop/tomatoproject/ssd-pytorch-master/img/2.jpg")


# #detect it
# result_red = find_strawberry_red(image)
# #write the new image
# cv2.imwrite('result_R.jpg', result_red)
# cv2.imshow("PPP", result_red)
# cv2.waitKey(0)



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



if __name__ == '__main__':
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
    # frame = Image.fromarray(np.uint8(frame))
    # 进行检测
    image,x,y,w,h = find_strawberry_red(frame)
    frame = np.array(image)
    temp = []
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    # print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("video",frame)
    # cv2.imshow('depth',color_data[3])
    cv2.waitKey(1)
    if informations != temp:
        print("realdata:")
        print(informations)
        i = 0
        for information in informations:
            print(i)
            print(information)#label y,x h w 
            # time.sleep(1)
            center_x=information[2]+information[4]/2
            center_y=information[1]+information[3]/2
            center_x= 1920/2
            center_y = 1080/2
            print(center_x,center_y)
            x,y = color_point_2_depth_point(a._kinect,_DepthSpacePoint,a._kinect._depth_frame_data,[int(center_x),int(center_y)])
            print(x,y)
            depthpoint = depth_point_2_world_point(a._kinect, _DepthSpacePoint, [x,y])
            print("depthpoint:")
            print(depthpoint)

            # depthpoint = a.map_color_point_to_depth_point([int(center_x),int(center_y)])
            # x,y,z = depth2xyz(depthpoint[0],depthpoint[1], color_data[2][depthpoint[0]][depthpoint[1]])
            # # x,y,z = depth2xyz(depthpoint[0],depthpoint[1], a._kinect._depth_frame_data[depthpoint[0]][depthpoint[1]])
            # print("depthpoint:")
            # print(depthpoint)
            # print(x,y,z)
            i+=1
            # time.sleep(1)
        # time.sleep(5)   



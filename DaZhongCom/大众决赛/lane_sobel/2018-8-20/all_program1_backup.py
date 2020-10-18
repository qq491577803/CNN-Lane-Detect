#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:36:40 2018

@author: fantastic
"""
# ! /usr/bin/python
# -*- coding: utf-8 -*-
import socket
import serial
import time
import threading
import math
import picamera
import time
import numpy as np
import cv2
import numpy as np
import io
from picamera.array import PiRGBArray 
from draw_line import auto_lane


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1);  # open named port at 9600,1s timeot
# sock = socket.socket(socket.AF_INET, socket.SOCK_STSREAM)
# server_address = ("localhost", 12422)
server_address = ("172.20.10.4", 12345)
# print "connecting to %s:%s" % server_address, port
sock.connect(server_address)

def client_to_sever(response):
    if response == '':
        print('mess is None')
    else:
        print "Sending '%s'" % response
        # onnetion,client_address = sock.accept()
        sock.send(response)

        recv = sock.recv(1024)
        print('secver_to_client_mess: ',recv)
    return recv



def client_to_ardiuno(recv):
    print recv
    if recv == '':
        print('1223')
        return ''
    else:
        ser.write(recv);  # writ a string to port
        response = ser.readall( );  # read a string from port
        print('ardiuno_to_client_mess: ',response)
        return response



#drive arduino
def drive_program(steer,speed,direc,bueezr):
    steer-=30
    framehead=chr(100)
    steer_byte= chr(steer)
    speed_byte=chr(speed)
    direc_byte=chr(direc)
    buzzer_byte=chr(bueezr)
    frameend=chr(101)
    send_buf=framehead+steer_byte+speed_byte+direc_byte+buzzer_byte+frameend
    #if send_buf  >= 128 :
       # pass
##        send_buf = chr(126)
    print 'message',framehead,speed_byte,direc_byte,buzzer_byte,frameend,send_buf
    ser.write(send_buf.encode())
    



if __name__ == "__main__":
    stream = io.BytesIO()
    camera = picamera.PiCamera()
    w,h = 640,360
    camera.resolution = (w,h)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera,size=(w,h))
    size = (w,h)
    first_recv = sock.recv(1024)
    # print('first_recv:',first_recv)
    error = 0  # camera piancha
    
    threshold = 10  #  yuzhi
    depth = 280    #biaoding depth
    steer = 90  # duo ji
    speed = 25  # motor speed
    direc = 1  # motor direction
    buzzer = 0
    a=1
    last_error=0
    
 
    for frame in camera.capture_continuous(rawCapture,format='bgr',use_video_port = True):
        image = frame.array
        start = time.time()
        image = frame.array
        start = time.time()
        image,error,Radius = auto_lane(image)
        #if error=="None"and abs(error)>60:
            #sss=60
        
        print('Time:',time.time()-start)
        print('offset: ',error,'R :',Radius)
        rawCapture.truncate(0)  
    
    
        while 1:
             a += 1
             print ('!!!!!!!!!',a)
             #image alogthrim start(output +-,error)           
            #image alogthrim start(output +-,error)
             if error=="None":#left lane lose
                 theta=90-15
                 data_ardiuno = drive_program(theta, speed, direc, buzzer)#
                 print "1",theta,speed                    
             elif abs(error)>=threshold:  #xunji alogthrim
                 if error<-90:                
                     theta=145
                 else:
                 
                     if error>0:   #rigth steer              
                         theta = 90-(int)(1.4*(math.atan2(error,depth)/math.pi*180)-2) #1.3duo ji zhuanjiao
                     else:            #left steer
                         theta = 90-(int)(0.7*(math.atan2(error,depth)/math.pi*180)-1) #0.8duo ji zhuanjiao  -1

                 if theta >= 157:
                    theta = 157
                 if theta<30:
                     theta=30
                 data_ardiuno = drive_program(theta, speed, direc, buzzer)#buzzer
                 print "2",theta,error
             
             else:#normal alogthrim
                 if error-last_error>0:
                     steer=75     #80
                     data_ardiuno = drive_program(steer, speed, direc, buzzer)
                 else:
                     steer=95
                     data_ardiuno = drive_program(steer, speed, direc, buzzer)
                 print "3",steer,speed
             if error!='None':                 
                 last_error=error
             else:
                 last_error=0
                            
             break
             
                # # break
                # recv = data_ardiuno = drive_program(theta, speed, direc, buzzer)
                # if recv == "s":
                #     data_ardiuno = drive_program(theta, 0, direc, buzzer)
        # # break
        #




# -*- coding:utf - 8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
class Polyfit:
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.left_x = None
        self.right_x = None
        self.left_y = None
        self.right_y = None
        self.ym_per_pix = 30 / 720  # y方向上每个像素对应的实际长度
        self.xm_per_pix = 3.7 / 700  # x方向。。。
        self.margin = 80

    def poly_fit_skip(self, img):
        """
        拟合车道线
        """
        # 检测是否存在车道线
        if self.left_fit is None or self.right_fit is None:
            return self.poly_fit_slide(img)        #
        nonzero = img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        margin = self.margin
        left_lane_inds = ((nonzero_x > (self.left_fit[0] * (nonzero_y**2) + self.left_fit[1] * nonzero_y + self.left_fit[2] - margin)) & (
            nonzero_x < (self.left_fit[0] * (nonzero_y**2) + self.left_fit[1] * nonzero_y + self.left_fit[2] + margin)))
        right_lane_inds = ((nonzero_x > (self.right_fit[0] * (nonzero_y**2) + self.right_fit[1] * nonzero_y + self.right_fit[2] - margin)) & (
            nonzero_x < (self.right_fit[0] * (nonzero_y**2) + self.right_fit[1] * nonzero_y + self.right_fit[2] + margin)))
        # 提取左右两车道的像素坐标
        self.left_x = nonzero_x[left_lane_inds]
        self.left_y = nonzero_y[left_lane_inds]
        self.right_x = nonzero_x[right_lane_inds]
        self.right_y = nonzero_y[right_lane_inds]
        # 拟合
        self.left_fit = np.polyfit(self.left_y, self.left_x, 2)
        self.right_fit = np.polyfit(self.right_y, self.right_x, 2)
        vars = {}
        vars['left_fit'] = self.left_fit
        vars['right_fit'] = self.right_fit
        vars['nonzero_x'] = nonzero_x
        vars['nonzero_y'] = nonzero_y
        vars['left_lane_inds'] = left_lane_inds
        vars['right_lane_inds'] = right_lane_inds

        return self.left_fit, self.right_fit, vars

    def poly_fit_slide(self, img):
        """
        沿图像下半部分的所有列取直方图，实现滑动窗口以查找并追踪直到图像顶部的行将二阶多项式拟合到每一行
        """
        #图像下半部分的直方图
        histogram = np.sum(img[img.shape[0]//2:, :], axis=0)
        out_img = np.dstack((img, img, img)) * 255
        midpoint = np.int(histogram.shape[0] / 2)
        left_x_base = np.argmax(histogram[:midpoint])
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint
        #滑动窗口的个数
        num_windows = 9
        window_height = np.int(img.shape[0] / num_windows)
        nonzero = img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        left_x_current = left_x_base
        right_x_current = right_x_base
        margin = self.margin
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []
        for window in range(num_windows):
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = left_x_current - margin
            win_xleft_high = left_x_current + margin
            win_xright_low = right_x_current - margin
            win_xright_high = right_x_current + margin
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)
            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (
                nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (
                nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > minpix:
                left_x_current = np.int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > minpix:
                right_x_current = np.int(np.mean(nonzero_x[good_right_inds]))
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        self.left_x = nonzero_x[left_lane_inds]
        self.left_y = nonzero_y[left_lane_inds]
        self.right_x = nonzero_x[right_lane_inds]
        self.right_y = nonzero_y[right_lane_inds]
        
        print('leftx: ',self.left_x.shape,'rightx: ',self.right_x.shape)
        
        if self.left_x.size ==0 and self.left_y.size ==0 and self.right_y.size == 0 and self.right_x.size == 0 or self.left_x.size <=1:
            self.left_fit, self.right_fit,vars = 'None','None',None
            print('There is no lane has been detected ```')
            return self.left_fit, self.right_fit, vars
        elif self.left_x.size >1 and self.right_x.size <=1 and self.left_y.size >1 or self.right_y.size <=1:
            print 'right lane detect has been failed !'
            self.right_fit,vars = 'None',None
            self.left_fit = np.polyfit(self.left_y, self.left_x, 2)            
            return self.left_fit, self.right_fit, vars
        
            
        else:
            print self.left_y.size,self.right_y.size
            self.left_fit = np.polyfit(self.left_y, self.left_x, 2)
            self.right_fit = np.polyfit(self.right_y, self.right_x, 2)
            vars = {}
            vars['left_fit'] = self.left_fit
            vars['right_fit'] = self.right_fit
            vars['nonzero_x'] = nonzero_x
            vars['nonzero_y'] = nonzero_y
            vars['left_lane_inds'] = left_lane_inds
            vars['right_lane_inds'] = right_lane_inds
            vars['out_img'] = out_img
            vars['margin'] = margin

            return self.left_fit, self.right_fit, vars


if __name__ =="__main__":
    pass
    # # img0 = cv2.imread("./test_image.jpg")
    # # img = warp(img0)
    #
    # #sobel
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    # abs_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # cv2.imwrite("sobel_x.jpg",abs_sobel)
    # cv2.imshow("sobel_x",abs_sobel)
    # abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # abs_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # cv2.imshow("sobel_y",abs_sobel)
    # cv2.imwrite("sobel_y.jpg",abs_sobel)
    # #abs
    # abs_thresh = abs_threshold(img, orient='x', thresh=(20, 100))
    # cv2.imshow("abs",abs_thresh*255)
    # cv2.imwrite('./abs.jpg',abs_thresh*255)
    # cv2.waitKey(0)
    # #mag
    # mag_thresh = mag_threshold(img, sobel_kernel=3, mag_thresh=(50, 100))
    # cv2.imshow("mag",mag_thresh*255)
    # cv2.imwrite('./mag.jpg',mag_thresh*255)
    # cv2.waitKey(0)
    # #dir
    # dir_thresh = dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.2))
    # cv2.imwrite("dir.jpg",dir_thresh)
    # cv2.imshow("dir", dir_thresh * 255)
    # cv2.waitKey(0)
    # #hls
    # hls_threshold = hls_threshold(img, thresh=(100, 255))
    # cv2.imshow("hls", hls_threshold * 255)
    # cv2.imwrite("hls.jpg",hls_threshold)
    # cv2.waitKey(0)
    #
    # #hsv
    # hsv_thresh = hsv_threshold(img, thresh=([20, 100, 100], [35, 255, 255], [0, 0, 230], [180, 25, 255]))
    # cv2.imshow("hsv", hsv_thresh * 255)
    # cv2.imwrite("hsv.jpg",abs_sobel)
    # cv2.waitKey(0)
    #
    # #combine
    # combine = combined_threshold(img)
    # cv2.imshow("combie",combine*255)
    # cv2.imwrite("./combinee.jpg",combine*255)
    # cv2.waitKey(0)
# -*-coding: utf-8 -*-
import cv2
import numpy as np

def perspective():
    src = np.float32([[40,138], [72,27], [240,27], [280,138]])
    dst = np.float32([[40, 180], [40, 0], [280, 0], [280,180]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M,Minv
def warp(img):
    M, Minv = perspective()
    img_size = (320, 180)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped

def dec_lane(mask):
    mask = cv2.resize(mask,dsize=(16,9))
    mask = (mask/255)
    line_point = mask[2]
    list = []
    for i in range(len(line_point) - 1):
        list.append(abs((int(line_point[i + 1]) - int(line_point[i]))))    
    counter = sum(list)
    print('The counter is ：',counter)
    if 6 < counter<20 :
        flag = 1
        print('Decrease lane has been detect ...')
    else:
        flag = 0
        print 'NO dec_lane'
    return flag

def rightangle(mask):
    mask = cv2.resize(mask,dsize=(16,9))
    mask = mask/255
    counter = 0
    for i in range(mask.shape[1]):
        single_list = mask[:,i]
        tem_list = []
        for j in range(len(single_list) - 1):
            tem_list.append(abs(int(single_list[j + 1] - single_list[j])))
        counter = sum(tem_list)+counter
    print("The counter is :",counter)
    
    return counter

def sobel(warped,thresh = (30,180)):
    gray = cv2.cvtColor(warped,cv2.COLOR_RGB2GRAY)
    sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,1))
    scal_sobel = np.uint8(255*sobel/np.max(sobel))
    mask = np.zeros_like(scal_sobel)
    mask[(scal_sobel >=thresh[0]) & (scal_sobel <=thresh[1])] = 1
    mask = mask *255
    
    return mask


if __name__ == '__main__':
    img = cv2.imread('./408.jpg')
    img = cv2.resize(img, dsize=(320, 180))    
    frame = warp(img)
    mask0 = sobel(frame)
    num = dec_lane(mask0)
    #rightangle(mask0)
# -*- coding:utf - 8 -*-
import cv2
import numpy as np
from lane_detection import Polyfit
import time
from time import sleep
from decraselane import dec_lane

def perspective():
    src = np.float32([[32,101], [59,13], [252,13], [288,101]])
    dst = np.float32([[32, 180], [32, 0], [288, 0], [288,180]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M,Minv

def warp(img):
    M,Minv = perspective()
    img_size = (320, 180)
    warped = cv2.warpPerspective(img, M, img_size)
    cv2.imwrite('resize.jpg',warped)   
    cv2.imshow('warp',warped)
    cv2.waitKey(1)
    return warped
def draw(img, warped, left_fit, right_fit,para=256,meter = 343):#para shi chedao xian kuandu de xiangsu geshu 
    M,Minv = perspective()
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # cv2.imshow("color",color_warp)
    plot_y = np.linspace(0, warped.shape[0] - 1, img.shape[0])
    cv2.imwrite('res1.jpg',warped)
    
    if left_fit !='None' and right_fit != 'None':
        left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
        
       
        
        
        right_fit_x = left_fit_x + para
        #right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]
        middle_fit_x = (left_fit_x+right_fit_x)/2
        y_axis = warped.shape[0]-2
        print y_axis,"..............:::::"
        point_x = left_fit[0]*y_axis**2 + left_fit[1]*y_axis + left_fit[2]
        print 'point_x:::',point_x
        print '0000.0.0.0.0.0.',y_axis
        left = left_fit[0]*y_axis**2 + left_fit[1]*y_axis + left_fit[2]
        #right = right_fit[0]*y_axis**2 + right_fit[1]*y_axis + right_fit[2]
        right = left + para
        print 'left:',left
        print 'right:',right
        
        
        
        #xielv
        print '-------------------------------------------------------------------'
        print y_axis 
        k1 = 2 * left_fit[0]*y_axis + left_fit[1]
        k2 = 2 * left_fit[0]*(y_axis-30) + left_fit[1]
        k_error = round(k1 - k2,2)
        print 'k_error: ',k_error,k1,k2
        
        lane_center = (left+right)/2
        img_center = warped.shape[1]/2
        print 'lane_center : ',lane_center
        print "img_center",img_center
        offset = round((img_center -lane_center)*meter/para,2)
        dy = 2*left_fit[0]*147 + left_fit[1]
        ddy = 2*left_fit[0]
        R = ((1+dy**2)**(3/2))/ddy*meter/para
        pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
        pts_middle = np.array([np.flipud(np.transpose(np.vstack([middle_fit_x, plot_y])))])
        PTS =np.hstack(pts_middle)
        pts = np.hstack((pts_left, pts_right))
        try:
            #cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0),)
            cv2.polylines(color_warp,np.int_([PTS]),isClosed=False,color=(255,150,0),thickness=8)
            cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0, 0,255),thickness=3)
            # cv2.imwrite("./lane_persp.jpg")
            # cv2.imshow('colormap',color_warp)
        except:
            pass
        # cv2.imwrite("./draw_lane.jpg",color_warp)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        try:
            #cv2.putText(result, str('Radius:') + str(round(R, 2)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 5), 1)
            cv2.putText(result, str('offset:') + str(round(offset, 2)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 5), 1)
            pass
        except:
            pass
    else:#left_fit !='None' and right_fit == 'None'
        
        left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
        #right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]
        right_fit_x = left_fit_x + para
        middle_fit_x = (left_fit_x+right_fit_x)/2
        y_axis = warped.shape[0]/2
        left = left_fit[0]*y_axis**2 + left_fit[1]*y_axis + left_fit[2]
        right = left+para
        lane_center = (left+right)/2
        print 'lane_center : ',lane_center
        img_center = warped.shape[1]/2
        print "img_center",img_center
        offset = round((img_center -lane_center)*meter/para,2)
        dy = 2*left_fit[0]*90 + left_fit[1]#147shi jisuan zhongxindian pianyidian de yzuobiao 
        ddy = 2*left_fit[0]
        R = ((1+dy**2)**(3/2))/ddy*meter/para  
        
        pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
        pts_middle = np.array([np.flipud(np.transpose(np.vstack([middle_fit_x, plot_y])))])
        PTS =np.hstack(pts_middle)
        pts = np.hstack((pts_left, pts_right))
        try:
            #cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0),)
            cv2.polylines(color_warp,np.int_([PTS]),isClosed=False,color=(255,150,0),thickness=8)
            cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0, 0,255),thickness=3)
            # cv2.imwrite("./lane_persp.jpg")
            # cv2.imshow('colormap',color_warp)
        except:
            pass
        # cv2.imwrite("./draw_lane.jpg",color_warp)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        try:
            #cv2.putText(result, str('Radius:') + str(round(R, 2)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 5), 1)
            cv2.putText(result, str('offset:') + str(round(offset, 2)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 5), 1)
            pass
        except:
            pass
    print '-------------------------'
    cv2.imwrite('./res.jpg',result)
    # cv2.imshow("img",result)
    # cv2.waitKey(0)
    return result,offset,R

def hsv(warped):
    lower_black = np.array([0,0,0])
    upper_black = np.array([180,255,70])
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_black, upper_black)
    return mask

def sobel(warped,thresh = (30,180)):
    gray = cv2.cvtColor(warped,cv2.COLOR_RGB2GRAY)
    sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0))
    scal_sobel = np.uint8(255*sobel/np.max(sobel))
    mask = np.zeros_like(scal_sobel)
    mask[(scal_sobel >=thresh[0]) & (scal_sobel <=thresh[1])] = 1
    mask = mask *255    
    return mask
def canny(warped,low,high):
    gray = cv2.cvtColor(warped,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    canny = cv2.Canny(gray,low,high)
    mask = canny * 255
    return mask
           
           
def auto_lane(img):
    polyfit = Polyfit()
    img = cv2.resize(img,dsize=(320,180))
    
    frame = warp(img)
    cv2.imshow('warped',frame)
    
    mask_flag = hsv(frame)
    dec_flag = dec_lane(mask_flag)
    print 'dec_lane flag = ',dec_flag    
    mask = sobel(frame)
    #mask = hsv(frame)
    #mask = canny(frame,50,150)
    #mask = cv2.GaussianBlur(frame,(5,5),0)    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    mask = cv2.erode(mask,kernel)
    mask = cv2.dilate(mask,kernel)           
    cv2.imshow('mask',mask)
    cv2.waitKey(1)
    
    # frame = combined_threshold(frame)
    try:        
        left_fit, right_fit, vars = polyfit.poly_fit_skip(mask)
        if left_fit =='None'  and right_fit == 'None':
            result,offset,Radius = img,'None',10000
            cv2.imshow('re', cv2.resize(img, dsize=(640, 320)))
            cv2.waitKey(1)
        elif left_fit !='None' and right_fit == 'None' or left_fit !='None' and right_fit != 'None' :
            result,offset,Radius = draw(img, mask, left_fit, right_fit)
            cv2.imshow('re',cv2.resize(result,dsize=(640,320)))
            cv2.waitKey(1)            
        else:
            print '-----1----unexcept error--------1----'
            result,offset,Radius = img,0,10000

    except:
        print '---------unexcept error------------'
        result,offset,Radius = img,0,10000
    '''offset --------------------'''
    if  offset == 'None':
        offset = 'None'
    else:
        offset = offset 
    return result,offset,Radius

if __name__ == "__main__":    
     polyfit = Polyfit()
     original_video = './test4.avi'
     cap = cv2.VideoCapture(original_video)
     n = 0
     while (1):
         n = n+1
         ret,img = cap.read()
         #cv2.imwrite('./tst3/' + str(n) + '.jpg',img)
         start = time.time()
         cv2.imshow('01',img)
         cv2.waitKey(1)
         result, offset, Radius = auto_lane(img)
         print('Time per px :',time.time() - start)
         print(offset,Radius)
'''
    polyfit = Polyfit()
    img = cv2.imread("./278.jpg")
    result, offset, Radius = auto_lane(img)
    print offset,Radius
'''
            
# -*- coding:utf-8 -*-
import picamera
#import picamera.array
import time
import numpy as np
import cv2
import numpy as np
import io
from picamera.array import PiRGBArray 

from draw_line import auto_lane

def cap_read_write(w,h,fps):
    stream = io.BytesIO()
    camera = picamera.PiCamera()
    camera.resolution = (w,h)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera,size=(w,h))
    size = (w,h)

    for frame in camera.capture_continuous(rawCapture,format='bgr',use_video_port = True):
        image = frame.array
        start = time.time()
        image,offset,Radius = auto_lane(image)
        print('Time:',time.time()-start)
        print('offset: ',offset,'R :',Radius)
        rawCapture.truncate(0)
        

if __name__ == '__main__':
    image,offset,Radius  = cap_read_write(640,360,10)

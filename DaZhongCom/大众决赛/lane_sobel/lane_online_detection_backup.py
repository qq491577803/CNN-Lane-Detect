# -*- coding:utf-8 -*-
import picamera
#import picamera.array
import time
import numpy as np
import cv2
import numpy as np
import io
from picamera.array import PiRGBArray 
from draw_line import au# -*- coding:utf-8 -*-
import picamera
#import picamera.array
import time
import numpy as np
import cv2
import numpy as np
import io
from picamera.array import PiRGBArray 
from draw_line import auto_lane


def cap_read_write(w,h,fps,long):    

    image = frame.array
    start = time.time()
    image,offset,Radius = auto_lane(image)
    print('Time:',time.time()-start)
    print('offset: ',offset,'R :',Radius)
    rawCapture.truncate(0)
        #return image,offset,Radius
    return image,offset,Radius

if __name__ == '__main__':
    #cap_read_write(640,360,10,8000)# -*- coding:utf-8 -*-
    stream = io.BytesIO()
    camera = picamera.PiCamera()
    camera.resolution = (w,h)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera,size=(w,h))
    size = (w,h)
    
    for frame in camera.capture_continuous(rawCapture,format='bgr',use_video_port = True):
        image,offset,Radius  = cap_read_write(640,360,10,8000)
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
server_address = ("192.168.191.2", 12340)
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
    framehead=chr(100)
    steer_byte= chr(steer)
    speed_byte=chr(speed)
    direc_byte=chr(direc)
    buzzer_byte=chr(bueezr)
    frameend=chr(101)
    send_buf=framehead+steer_byte+direc_byte+buzzer_byte+frameend
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
    error = 0  # bianliang
    threshold = 5  # const
    depth = 460
    steer = 90  # duo ji
    speed = 20  # motor speed
    direc = 1  # motor direction
    buzzer = 0
    a=1
    for frame in camera.capture_continuous(rawCapture,format='bgr',use_video_port = True):
        image = frame.array
        start = time.time()
        image = frame.array
        start = time.time()
        image,error,Radius = auto_lane(image)
        if error=="None":
            error=0
        print('Time:',time.time()-start)
        print('offset: ',error,'R :',Radius)
        rawCapture.truncate(0)  
    
    
        while 1:
             a += 1
             print (a)
             #image alogthrim start(output +-,error)           
            #image alogthrim start(output +-,error)
             if abs(error)>=threshold:  #xunji alogthrim
                 theta = 90-(int)(math.atan2(error,depth)/math.pi*180) #duo ji zhuanjiao
                 # theta >= 127:
                     #theta = 127
                 data_ardiuno = drive_program(140, speed, direc, buzzer)
                 print str('jiaodu'),theta,error
             else:#normal alogthrim
                 data_ardiuno = drive_program(130, speed, direc, buzzer)
                 print steer,speed,direc,buzzer             
             break
             
                # # break
                # recv = data_ardiuno = drive_program(theta, speed, direc, buzzer)
                # if recv == "s":
                #     data_ardiuno = drive_program(theta, 0, direc, buzzer)
        # # break
        #





# -*- coding:utf-8 -*-
import cv2
import numpy as np
import time
from glob import  glob


lower_green = np.array([35,43,46])
upper_green = np.array([77,255,255])
lower_cheng = np.array([11,43,46])
upper_cheng = np.array([25,255,255])
lower_lan = np.array([78,43,46])
upper_lan = np.array([124,255,255])
red = {'low':np.array([156,43,46]),'upper':np.array([180,255,255])}
green = {'low':np.array([35,43,46]),'upper':np.array([77,255,255])}
cheng = {'low':np.array([11,150,160]),'upper':np.array([25,255,230])}
blue = {'low':np.array([78,130,46]),'upper':np.array([124,255,255])}
dict = {'green':green,'cheng':cheng,'blue':blue,'red':red}
def recongise(BGR):
    hsv = cv2.cvtColor(BGR,cv2.COLOR_BGR2HSV)
    name = 'None'
    location = 'None'
    for key in dict:
        color =  dict[key]
        mask = cv2.inRange(hsv,color["low"],color['upper'])
        mask = cv2.erode(mask,kernel=(4,4),iterations=4)
        mask = cv2.dilate(mask,kernel=(4,4),iterations=4)
        cv2.imshow('mask',mask)
        cv2.waitKey(1)
        #轮廓检测
        cnts= cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        # binary, cnts,hier= cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        try:
            #找出最大轮廓
            c = max(cnts,key=cv2.contourArea)
            #确定轮廓的外接圆
            ((x,y),radius) = cv2.minEnclosingCircle(c)
            # cv2.drawContours(img,c,-1,(0,0,0),3)
            M = cv2.moments(c)
            #轮廓中心坐标
            center = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
            # cv2.circle(img,(int(x),int(y)),int(radius),(0,255,255),2)
            cv2.circle(img,center,5,(0,0,0),-1)
            name = key
            location = center
            # ((x,y),radius) = cv2.minEnclosingCircle()
            #cv2.imshow('img',img)
            # cv2.imshow('img',mask)
            #
            #cv2.waitKey(0)
            #print(name)
        except:
            # print('Threr is no light has been found !!')
            pass
    print "Traffic:",'name:',name,'location:',location
    return name,location

if __name__ == '__main__':
    fpts =  glob('./pic/light.jpg')
    for fn in fpts:
        print(fn)
        img = cv2.imread(fn)
        img = cv2.resize(img, dsize=(640, 360))
        start = time.time()
        name,location = recongise(img)
        print('Ttime:',time.time() - start)
        print(name,location)
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
            print '000000000000000'
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
# -*- coding:utf - 8 -*-
import cv2
import numpy as np
from lane_detection import Polyfit
import time
from time import sleep

def perspective():
    src = np.float32([[40,138], [72,27], [240,27], [280,138]])
    dst = np.float32([[40, 180], [40, 0], [280, 0], [280,180]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M,Minv

def warp(img):
    M,Minv = perspective()
    img_size = (320, 180)
    warped = cv2.warpPerspective(img, M, img_size)
    cv2.imshow('warp',warped)
    cv2.waitKey(1)
    
    return warped
def draw(img, warped, left_fit, right_fit,para,meter):#para shi chedao xian kuandu de xiangsu geshu 
    M,Minv = perspective()
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # cv2.imshow("color",color_warp)
    plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    
    
    if left_fit !='None' and right_fit != 'None':
        left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
        right_fit_x = left_fit_x + para
        #right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]
        middle_fit_x = (left_fit_x+right_fit_x)/2
        y_axis = warped.shape[0]/2
        left = left_fit[0]*y_axis**2 + left_fit[1]*y_axis + left_fit[2]
        #right = right_fit[0]*y_axis**2 + right_fit[1]*y_axis + right_fit[2]
        right = left + para
        lane_center = (left+right)/2
        img_center = warped.shape[1]/2
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
        img_center = warped.shape[1]/2
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
    # cv2.imshow("img",result)
    # cv2.waitKey(0)
    return result,offset,R
def auto_lane(img):
    polyfit = Polyfit()
    img = cv2.resize(img,dsize=(320,180))
    lower_black = np.array([0,0,0])
    upper_black = np.array([180,255,70])
    frame = warp(img)
    cv2.imshow('frame',frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_black, upper_black)
    cv2.imwrite('./mask.jpg',mask)
    cv2.imshow('frame:',mask)
    cv2.waitKey(1)
    
    # frame = combined_threshold(frame)
    try:        
        left_fit, right_fit, vars = polyfit.poly_fit_skip(mask)
        if left_fit =='None'  and right_fit == 'None':
            result,offset,Radius = img,0,10000
            cv2.imshow('re', cv2.resize(img, dsize=(640, 320)))
            cv2.waitKey(1)
        elif left_fit !='None' and right_fit == 'None' or left_fit !='None' and right_fit != 'None' :
            result,offset,Radius = draw(img, mask, left_fit, right_fit,para = 277,meter = 30)
            cv2.imshow('re',cv2.resize(result,dsize=(640,320)))
            cv2.waitKey(1)            
        else:
            print '-----1----unexcept error--------1----'
            result,offset,Radius = img,0,10000

    except:
        print '---------unexcept error------------'
        result,offset,Radius = img,0,10000
        
    return result,offset,Radius

if __name__ == "__main__":
    '''
     polyfit = Polyfit()
     original_video = './test.avi'
     cap = cv2.VideoCapture(original_video)
     n = 0
     while (1):
         n = n+1
         ret,img = cap.read()
         #cv2.imwrite('./tst2/' + str(n) + '.jpg',img)
         start = time.time()
         cv2.imshow('01',img)
         cv2.waitKey(1)
         result, offset, Radius = auto_lane(img)
         print('Time per px :',time.time() - start)
         print(offset,Radius)
    
'''
    polyfit = Polyfit()
    img = cv2.imread("./408.jpg")
    result, offset, Radius = auto_lane(img)
    print offset,Radius

            
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
import thread


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1);  # open named port at 9600,1s timeot
# sock = socket.socket(socket.AF_INET, socket.SOCK_STSREAM)
# server_address = ("localhost", 12422)
server_address = ("192.168.31.204", 12345)
# print "connecting to %s:%s" % server_address, port
sock.connect(server_address)
#duoxiancheng
def receivedata_from_server():
    data_recv  = sock.recv(1024)
    a = 'Have received:'+data_recv
    ser.write(a)



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
    framehead=chr(100)
    steer_byte=chr(steer)
    speed_byte=chr(speed)
    direc_byte=chr(direc)
    buzzer_byte=chr(bueezr)
    frameend=chr(101)
    send_buf=framehead+steer_byte+speed_byte+direc_byte+buzzer_byte+frameend
    if send_buf  >= 128 :
        pass
##        send_buf = chr(126)
    ser.write(send_buf.encode())
    



if __name__ == "__main__":
    global data_recv
    thread.start_new_thread(receivedata_from_server,())
    stream = io.BytesIO()
    camera = picamera.PiCamera()
    w,h = 640,360
    camera.resolution = (w,h)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera,size=(w,h))
    size = (w,h)
    first_recv = sock.recv(1024)
    # print('first_recv:',first_recv)
    error = -30  # bianliang
    threshold = 20  # const
    depth = 50
    steer = 90  # duo ji
    speed = 20  # motor speed
    direc = 1  # motor direction
    buzzer = 0
    a=1
    for frame in camera.capture_continuous(rawCapture,format='bgr',use_video_port = True):
        image = frame.array
        start = time.time()
        image = frame.array
        start = time.time()
        image,error,Radius = auto_lane(image)
        print('Time:',time.time()-start)
        print('offset: ',error,'R :',Radius)
        rawCapture.truncate(0)  
    
    
        while 1:
             a += 1
             print (a)
             if data_recv=='s': 
                 if abs(error)>=threshold:  #xunji alogthrim
                     theta = 90-(int)(math.atan2(error,depth)/math.pi*180) #duo ji zhuanjiao
                     if theta >= 127:
                         theta = 127
                     data_ardiuno = drive_program(theta, speed, direc, buzzer)
                     print '111'
                 else:#normal alogthrim
                     print '222'
                     data_ardiuno = drive_program(steer, speed, direc, buzzer)
                     print steer,speed,direc,buzzer
                 break
             elif data_recv =='n':
                steer = 90
                speed = 0
                direc = 1
                buzzer = 1
                data_ardiuno =  drive_program(steer,speed,direc,buzzer)
                # # break
                # recv = data_ardiuno = drive_program(theta, speed, direc, buzzer)
                # if recv == "s":
                #     data_ardiuno = drive_program(theta, 0, direc, buzzer)
        # # break
        #





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
from light import recongise

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1);  # open named port at 9600,1s timeot
# sock = socket.socket(socket.AF_INET, socket.SOCK_STSREAM)
# server_address = ("localhost", 12422)
server_address = ("192.168.8.102", 12330)
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
    framehead=chr(100)
    steer_byte= chr(steer)
    speed_byte=chr(speed)
    direc_byte=chr(direc)
    buzzer_byte=chr(bueezr)
    frameend=chr(101)
    send_buf=framehead+steer_byte+direc_byte+buzzer_byte+frameend
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
    error = 0  # bianliang
    threshold = 5  # const
    depth = 460
    steer = 90  # duo ji
    speed = 20  # motor speed
    direc = 1  # motor direction
    buzzer = 0
    a=1
    '''light_para'''
    lower_green = np.array([35,43,46])
    upper_green = np.array([77,255,255])
    lower_cheng = np.array([11,43,46])
    upper_cheng = np.array([25,255,255])
    lower_lan = np.array([78,43,46])
    upper_lan = np.array([124,255,255])
    red = {'low':np.array([156,43,46]),'upper':np.array([180,255,255])}
    green = {'low':np.array([35,43,46]),'upper':np.array([77,255,255])}
    cheng = {'low':np.array([11,150,160]),'upper':np.array([25,255,230])}
    blue = {'low':np.array([78,130,46]),'upper':np.array([124,255,255])}
    dict = {'green':green,'cheng':cheng,'blue':blue,'red':red}
    
    
    for frame in camera.capture_continuous(rawCapture,format='bgr',use_video_port = True):
        start = time.time()
        stream = frame.array
        start = time.time()
        image,error,Radius = auto_lane(stream)
        name,location = recongise(stream)
        print "Traffic:",'name:',name,'location:',location
        if error=="None":
            error=0
        print('Time:',time.time()-start)
        print('offset: ',error,'R :',Radius)
        rawCapture.truncate(0)  
    
    
        while 1:
             a += 1
             print (a)
             #image alogthrim start(output +-,error)           
            #image alogthrim start(output +-,error)
             if abs(error)>=threshold:  #xunji alogthrim
                 theta = 90-(int)(math.atan2(error,depth)/math.pi*180) #duo ji zhuanjiao
                 if theta  > 127:
                     theta = 127
                 data_ardiuno = drive_program(theta, speed, direc, buzzer)
                 print str('jiaodu'),theta,error
             else:#normal alogthrim
                 data_ardiuno = drive_program(90, speed, direc, buzzer)
                 print steer,speed,direc,buzzer             
             break
             
                # # break
                # recv = data_ardiuno = drive_program(theta, speed, direc, buzzer)
                # if recv == "s":
                #     data_ardiuno = drive_program(theta, 0, direc, buzzer)
        # # break
        #




���� JFIF      �� C 			





	


��  �@ ��           	
�� �   } !1AQa"q2���#B��R��$3br�	
%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz���������������������������������������������������������������������������   ? ������� ���s� �g� Q�&� ����� �� ���I� ��~� Q^�X�� ����>%���� a� �����c�o�?i���}�ֿh�w���~n���v�n\�tQE~� ���t_�$� �~�����:(��� � �\�:� ?�f� �cV����+�� ����:� � �Y� �cI��(��k���� ��*�m���e������_��xoP��WR]j��7V��(bTѮ��u`�B\3������:(���� (���/�<��~�����Ǉ�3��]�� ��v���"�Q�]OV�[��@3J-���a&v���!E��� 1� ��ܓ� ������� 袊��� �� ���I� ��~� QE tQE}� � �� �u���� �ƭ_��W�e� ���⯏��X�'�^1���k�����L��3k�8�-\�w>k[���B�ʪ)�(QE~� ���t_�$� �~�����:(����	��?
���
I�>x�^���Z���;gFլ�����mb�9m�@RX����VV �q�Q_���� )M������ ��r� h��k���� �~*пm���������⯅�x�P��'7Q�iZ�6��"d�n��Fb��U�+������:(�����?��/���?_��Q_�Q_� ��?����3�1�W��E�A�X��)���������y^E��� 1� ��ܓ� �������
�� (�������e� �;�� ����� �$���7�i� ���o��,��}���|��O	�V��_�O���3�ŭ�:�l5f;$[��[g�$���P7�Q=�Ӣ1*�s1P���(����2��R����7�W� O�O�Q_���<'�x�S�/��3�h�ދ�Ma��ڵ��]X]B�9m�@)QՑ��ee �Eg�E~���?��ٿ��|���~��� 袊�� �q� ���� ��� Q�Z���(���
�� )M�����<e� ����
(��{�t�[�#�g��Pм�x���4�O�Q�_��j�C�2'��bۙB���,Q���Q^� b� �Y~���o�2� ��TQE}� � �� �u���� �ƭ_��W����/��c��/� 	O�� ��G�؟a����{u��;x����M�66� ��ryx��+�����g�|9� _�F��i�z� ��^�B���H��n�S�n��bҋ{k��&XG����Q�Q_����R��K��x�� O���Q_���eO��7���o����4:������:(��� � �\�:� ?�f� �cV����+��� ���Sio�8����+�~� �*�^>� ����ǡ�����;��_ ��������7w���sG�����o�������Ҽd��EB��
7�EW�� ���?඿<Y��z��?�-o�7j�?�XI���6�`�f�P��la�yۈ؎��n�E~H���W�� ���<=�B��@��_�VVO,zu�i����N��#qsow'�3���ʊ+�� �5��S��3��o�F5j��h���/�+��7��� �����/+�(��_��ʟ�Jo�����_�>hu�?QE tQE~�� ��� |w��k��Şоק���"��Sq������MΚ�mv&n�8����������W�� b� ������p2� ���xW��� T� �,�}� g�� �C���+��:;�PQ�����I���@����~� �*�)�>� ��������E��� ((��� r����M TQ_� ��?����3�1�W��E�A�X��)���������y^E��� T� �S|}� f� �� ��C����+���+����� ������� U� ��_��W�G�g◎�8�S�7ƿ�Z�����a��o�z��b��w�s��yp�G�dvڊ�� ��EU��g��O|+� �?G��|֟�(k~$У���H-bK]!�pȡe�F�p�!u��#�1dO��+�C�����_��E�x�:~�5�įx�z�Q#G�Wɫ��YԬ_gҮ�c�6�,���QE��� >�������y�j��W��/���X�Ak%��usup�)B�U}�!WU$���)O躊+��/����h��#o�	��J�� ��������g�$�k?g۽1�}��ߓ�����ݍ�aQE}?� R��㿃� �W��ş����B����\\}�)��jwi��C�Uee���[�ܞf�*�?��(��K��)�_�?�ZO����5�K� E�I���G�]k66ڽ����[��R0Aa�fv��
(��� �47��*O�,����E��?�h|?׼1���y_ٞT)���n�����l�c��/c� W�Q_�Q_�� �f�<+���*� �4��O���������E&�t���t��+f�[�\LQ2�8%|mF#����� 袊�����AG������'ի��+���_�g�l� ��<y�m>-oQ��i���^��]Z�h��w�N��'��GuQ�aA�s���E���~�U��io^xgP�D�u	Xi�̖N��WVɫ�qo�lybK�WtRY��	?��Q^� b� �Y~���o�2� ��TQ^�� ����O�߷��� ����x?��u��e��XZj���M�¯$�"��j+3c
	 W�9E�Q�i�i�k��*��o��ڷ��>��ޕ���ΰ�4�0.�aw�I_:��	ZDm��3 TeQ�W�� �@��_�g�� z�%���^����O]�Z����:���w�Gwq,�GV��y�H��C�
��+�� j��_�g�o�8�����׫}����ƫ������v��|���3�c���̚h�����(��� ��ߴ�����
��K�ƽ[�;�� �ҵ=Q�t� �=6�L���Y�8����9f���9*� g�W��[_�)����&'�;��?� �>#x������wM������]kb�ݥ�����%��PbUڄI$��/�EW�w��� S�5�/Q�?�S㷈�� K�� E��0��1Ǩ�3[�is�Or��IeK[#h�Z�$�ZYm�躳�Y��
�º���u�m?E�4]>k�gYկR����2Kq4��D����B��$�+�� ���?����$­־��o��4��v��+��=-�ߩM��������W��[�o�fi��EW��� T� �,�}� g�� �C�����,�R�'�&���K]���ǃ�?{���S�,�������o.y$�n�QY�PI�
�� h_����>x��s�֡�Zh�?�~��f�I�#�����KY^�P��R��U��3��+�;�� F|3���xG�� ���څ��<)���xk�7W���v�|�=6��#�v�0f��krV(���*$��_� b� �Y�K?g� �<�� %W�������Ox��K� �m����^����j�4�^���/)�2i�F�&�r"����l֓G-�2D��QE~��� ���U������ C���<A�����f�/�oű��:��K�&&�yo�y.%X��y�I���g�O� 9��u?�� �Ρ���i�\Z�ZO�}}.�-�-ue,�S,����E���.?�w��
��'�����:�Ï��Z��w�K�	u��5�ƤѨH�DXd[icK��<�;v��B�(��S� ����uw�q� ��5����i� |	��>KoG���]3Y�#�"A}�\$�0�L���J���0G �ş�|���<+����	����ܺ|ɣj�ŧ����(DR�o����h�h��!;����� ��?�o~3~�5jW�..4/Eu*h��a5��f��k� V#2JaY&ye-#x}QE}���C� ����%�ï	��� ����7��)����_��+��m��i=��g!|�[b�Eڈ�<C��� ��~����=��q�Ux�Ɩ�j_���u#�mu����2�Ggm(��K�
3ef%�r�EQ_�o���Q�q� ~��7��/�P���3{momik��Es+��5�i��A �D-�q?�}x���ڏ�� ����t{m>k�'��/maմ�{�W�-��f���H.b%@heG��*��H>�E tQEQEQEQEQEQEQEQEQ_����� ((�� s7����}� _(�t�#���_�����?���.���_���iZaq$��>�P;"��Vv�2�ɕ�ھ�s"}_E�����e� ڟ�_��� 	O���+��φ?����o��_Mk��'|�W��oٽ�����O��EQEQEQEQEQEQEQEQ_���K����� ����5���������j_j���։}�D}�����f�������7�������g�|9� 9�������}�޿�xn�B���H��n�_��kx�3J-�&(�a�6��E_�V/�Jo�-� g�/�>^W�QEQEQEQEQEQEQEQEQ_�g�'�~*��o�|�/��3�h�ދ�?�v΍�Y=�Յ�:=�r���<R��## ��A ��
��� ��|M��?�J�v6�{ŭ|p�,��I���8�LծA$���˾hJHcib,b�T��+��� ��~
xW������/��B��� �x�i59Q�Z͍��t�� ���T�XF���v�(��(��(��(��(��(��(��(���� (���=��^;���~xOO�u��^�`[�_ſ��'�m�U�Y����2o��q}/��,_��X:c�z��@�����N��?����k�
(��(��(��(��(��(��(��(���>�-����~�)�B���?�<Ae�xsL�TP}���t��2fH��,����W9b &���+���� �]� ��� p��E*_�x_�O
�+� ����j����$�c��8.�{�!R ��b�>�n�9v�S�)TO�
(��(��(��(��(��(��(��(���� �$���7�i� ���o��,��}����>�_�uߏ� �g;=?P]o¾ּI�]IY-u[�k[t���2���E�"�Y!*�Y�E�W�� ��S|� f� ���\��(��(��(��(��(��(��(��(���� ��/�#�%�=������ ����a�G��������Ly�`�7��~n���cg�W�� ��S|� f� ���\�����|��O�־*��|cΠn��� �O���ڬp>���ڴ�e&�@�F�$�����!?h��(��(��(��(��(��(��(��+�� �5��S��3��o�F5j��h��K���g��G� �������څ�����n�B���ycӭ[@��kx��n.n&(�S$�7;�v�_�?�|���Uy�_٧�V~�%�4�CŶ��������i[��(YR����u���"6���EQEQEQEQEQEQEQEQ_�� �f��+�?�*� �5��gO���>�����K&�tڞ�j�3a����B]0�9�L�v�����/�9�.��G�k����\� ��_�;�ͧ�6V�s��[g>[8�,�s��,1� c�W����?
��6�x���:|�ޝ����O�d�F������Y.-㔍��kj�B�[BX�����(��(��(��(��(��(��(��(����2_�o�����ƽ?B�<1�� ��j��ڢg��5[����	̋L�mʥW��.����+��� ���Sio�8�����&O�<U���&����x�Pֵ�k�����gV�{�����{Y%��Y	yewfvv%���I5��� ��,�� g��\���(��(��(��(��(��(��(��(�����?��/���?_��W�G�g◎�8�S�7ƿ�Z�����a��o�z��b��w�s��yp�G�dvڊ�� ��� �<M�x��	A�6�]��Q|�œ.���YHd��-��a�h�x��09�),L�H��G���tw������?���5��QEQEQEQEQEQEQEQEW��� G����d��,�A��|P��z�������Jӡ��x� q+>�t������X��us� ~)x�w��|k���� e�c�����G���Y��ai�q7�
��l�7m��͌($�_�_��k�����g���������|� G�
>9� ܳ� �>�_�QEQEQEQEQEQEQEQEO��eO������p����:�~� � ���,�io�7� ��*��import numpy as np
a = 1.580
b = 1
c = a - b
print c
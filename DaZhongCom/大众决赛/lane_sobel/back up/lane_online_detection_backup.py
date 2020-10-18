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
        #è½®å»“æ£€æµ‹
        cnts= cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        # binary, cnts,hier= cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        try:
            #æ‰¾å‡ºæœ€å¤§è½®å»“
            c = max(cnts,key=cv2.contourArea)
            #ç¡®å®šè½®å»“çš„å¤–æ¥åœ†
            ((x,y),radius) = cv2.minEnclosingCircle(c)
            # cv2.drawContours(img,c,-1,(0,0,0),3)
            M = cv2.moments(c)
            #è½®å»“ä¸­å¿ƒåæ ‡
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
        self.ym_per_pix = 30 / 720  # yæ–¹å‘ä¸Šæ¯ä¸ªåƒç´ å¯¹åº”çš„å®é™…é•¿åº¦
        self.xm_per_pix = 3.7 / 700  # xæ–¹å‘ã€‚ã€‚ã€‚
        self.margin = 80

    def poly_fit_skip(self, img):
        """
        æ‹Ÿåˆè½¦é“çº¿
        """
        # æ£€æµ‹æ˜¯å¦å­˜åœ¨è½¦é“çº¿
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
        # æå–å·¦å³ä¸¤è½¦é“çš„åƒç´ åæ ‡
        self.left_x = nonzero_x[left_lane_inds]
        self.left_y = nonzero_y[left_lane_inds]
        self.right_x = nonzero_x[right_lane_inds]
        self.right_y = nonzero_y[right_lane_inds]
        # æ‹Ÿåˆ
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
        æ²¿å›¾åƒä¸‹åŠéƒ¨åˆ†çš„æ‰€æœ‰åˆ—å–ç›´æ–¹å›¾ï¼Œå®ç°æ»‘åŠ¨çª—å£ä»¥æŸ¥æ‰¾å¹¶è¿½è¸ªç›´åˆ°å›¾åƒé¡¶éƒ¨çš„è¡Œå°†äºŒé˜¶å¤šé¡¹å¼æ‹Ÿåˆåˆ°æ¯ä¸€è¡Œ
        """
        #å›¾åƒä¸‹åŠéƒ¨åˆ†çš„ç›´æ–¹å›¾
        histogram = np.sum(img[img.shape[0]//2:, :], axis=0)
        out_img = np.dstack((img, img, img)) * 255
        midpoint = np.int(histogram.shape[0] / 2)
        left_x_base = np.argmax(histogram[:midpoint])
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint
        #æ»‘åŠ¨çª—å£çš„ä¸ªæ•°
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




ÿØÿà JFIF      ÿÛ C 			





	


ÿÀ  ´@ ÿÄ           	
ÿÄ µ   } !1AQa"q2‘¡#B±ÁRÑğ$3br‚	
%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyzƒ„…†‡ˆ‰Š’“”•–—˜™š¢£¤¥¦§¨©ª²³´µ¶·¸¹ºÂÃÄÅÆÇÈÉÊÒÓÔÕÖ×ØÙÚáâãäåæçèéêñòóôõö÷øùúÿÚ   ? ıü¯äşÿ ”ëüsÿ ¹gÿ Q&¾ ¢Šışÿ ƒÿ æè¿îIÿ Üı~ÿ Q^ûXüÿ † ı–>%şÍğ”ÿ aÿ ÂÄø¬øcûoì?işÏû}ŒÖ¿hòwÇæù~nı›ÓvÜn\ätQE~ÿ ÁŒót_÷$ÿ î~¿¨¢¿€:(¢¾ÿ ÿ ƒ\å:ÿ ?îfÿ ÔcV¯ëöŠ+ùÿ ƒ£¿å:ÿ ÿ îYÿ ÔcI¯€(¢¿k¿àÈÿ ‚*×mŒŸ´e¡§®‰á_…ğxoPµ’WR]jºŒ7Vï„(bTÑ®ƒ–u`ÒB\3ş“¨¢¿€:(¢Šşÿ (¢¿—/ø<÷Â~ğçüÂúÇ‡¼3§Ø]ëÿ ô‹ıvêÊÉ"“Qº]OVµ[‰Ù@3J-í­áùa&v¢ù!Eûıÿ 1ÿ ÍÑÜ“ÿ ¹úış¢Šş è¢Šışÿ ƒÿ æè¿îIÿ Üı~ÿ QE tQE}ÿ ÿ ¸ÿ ÊuşÜÍÿ ¨Æ­_×íWñeÿ Òø×â¯ßğXÚ'Ç^1Óôûk»Š†áL‰Ò3k£8Ò-\‡w>k[ØÄòB™Êª)¿(QE~ÿ ÁŒót_÷$ÿ î~¿¨¢¿€:(¢½Ãş	“á?
ø÷ş
Iû>xÇ^Óõ­Zøáá;gFÕ¬’æÖşÖmbÖ9mæŠ@RX‘‘VV ‚q”Q_Ìü­ÿ )Mğı›ş•ÿ §Ír¿ h¢¿k¿àÈÿ ~*Ğ¿mŒŸ³Ÿ§¶‰â¯…ğx“Pº’'7QİiZŒ6¶é"dÖn‹†FbÑÂU+ş“¨¢¿€:(¢¿¿àÆ?ùº/û’÷?_¿ÔQ_ÀQ_ÿ Á®?òŸ÷3ê1«WõûEüAÁX¿å)¿´·ıœŒ¿ôùy^Eûıÿ 1ÿ ÍÑÜ“ÿ ¹úış¢¼ş
Åÿ (²ı¥¿ìßüeÿ ¦;Êş ¨¢Š÷ÿ ø$ïü¥7öiÿ ³€ğoşŸ,ëû}¢Šş|àùO	øVÏÅ_³O¬ü3§Å­ê:‹l5f;$[««[gÒ$··’P7¼Q=ÕÓ¢1*s1PŸÁ(¯×ïø2§şR›ãïû7ıWÿ OšOÔQ_ÀŸ‹<'â¯x«Sğ/¼3¨hºŞ‹¨Ma¬èÚµ“Û]X]Bæ9mæŠ@)QÕ‘‘€ee €EgÑE~¿Á•?ò”ßÙ¿ê¿ú|Ğëú~¢Šş è¢Šûÿ şqÿ ”ëüÿ ¹›ÿ QZ¿¯Ú(¯âş
Åÿ )Mı¥¿ìà<eÿ §ËÊğ
(¯è{ştø[ã½#ágíñ¯PĞ¼¿xƒÄÑ4OíQ´_éğjŞCå†2'—§bÛ™B·Ÿ…,QÂşïQ^ÿ bÿ ”Y~Òßöoş2ÿ ÓåTQE}ÿ ÿ ¸ÿ ÊuşÜÍÿ ¨Æ­_×íWâüóğ/şÙcàí/ÿ 	O“ÿ —ÄGÃØŸaİö¿í{uö;xòü¯ìM›66ÿ ´çryxçŠ+õ¿şÂñg…|9ÿ _ñFâiözÿ Àı^ÃBµ½½H¤Ôn—SÒnšŞbÒ‹{k‰Š&XG¯¨ÄQ´Q_Äü‹şR›ûKÙÀxËÿ O—•àQ_¯ßğeOü¥7Çßöoú¯şŸ4:şŸ¨¢¿€:(¢¾ÿ ÿ ƒ\å:ÿ ?îfÿ ÔcV¯ëöŠ+ø‚ÿ ‚±ÊSioû8éòò¼Š+ú~ÿ ƒ*å^>ÿ ³€ÕôÇ¡×ëõñ‡ü;ñ¯Å_ à‹´¼§é÷7wş‹ÃsG©ÄïµÖo­´‹§š¶÷Ò¼d’¢EBÊê
7ñ¥EWèÿ üğ·Ç?à¶¿<Yáíz¼?â-oÅ7jŠ?±XI¤Üé©6×`ÒfïP³‹la˜yÛˆØËınÑE~HÁç¾ñWˆÿ à”Ö<=áBşÓ@øá¤_ë·VVO,zu«išµªÜNÊ†#qsowÂ™'‰3¹ÔåÊŠ+ïÿ ø5ÇşS¯ğ3şæoıF5jş¿h¢¿ˆ/ø+ü¥7ö–ÿ ³€ñ—şŸ/+À(¢¿_¿àÊŸùJo¿ìßõ_ı>huı?QE tQE~ÿ Á§ÿ |wñşkğãÅĞ¾×§øÃş"ÖüSqö¨£û„šMÎš“mv&nõ8¶Æ‡¸ˆì¿ÖíWñÿ bÿ ”¦şÒßöp2ÿ ÓååxWôıÿ Tÿ Ê,¼}ÿ gªÿ éC¯×ê+àø:;şPQñÏşåŸıIôšş@¨¢Šı~ÿ ƒ*å)¾>ÿ ³Õôù¡×ôıEğüÿ ((øçÿ rÏş¤úM TQ_ÿ Á®?òŸ÷3ê1«WõûEüAÁX¿å)¿´·ıœŒ¿ôùy^Eúıÿ Tÿ ÊS|}ÿ fÿ ªÿ éóC¯éúŠ+ø¢Š+õûş©ÿ ”¦øûşÍÿ Uÿ Óæ‡_ÓõWğGñgâ—ş8üSñ7Æ¿ŠZïö§‰üaâİoÄzŸÙbƒíw÷s¼÷ypªGùdvÚŠª¹Â€ ÏÑEU¿ğg§ÁO|+ÿ ‚?Gã¯ê|ÖŸ¾(k~$Ğ£²•ÚH-bK]!’pÈ¡eûF•pà!uòŞ#¸1dOÔú+óCşÙø×á_…ğE¿xÄ:~¡5ßÄ¯xÃz–Q#GÔWÉ«³ÎYÔ¬_gÒ®ŞcÄ6…,éü™QEûÿ >üğ®»ñ÷ãïíy¨j­øWÁú/†ôûXåAk%®«usupò)BæU}Ô!WU$Á•Ë)OèºŠ+äø/ÇÀ¿øhø#oí	àøJ±ÿ ³şÍâµı‡ígö$Ñk?gÛ½1ç}ƒÈß“åù»ö¾İüaQE}?ÿ Rø¥ã¿ƒÿ ğWÙËÅŸµßìíBïã‡¢\\}–)·Øjwi¦ßC¶Uee¥İÄ[€Üfä*ê¬?µÚ(¯ãKşø)á_€?ğZOÚÀ¾Ô5›Kÿ EâI¤ÔåG]k66Ú½ÒˆƒÊ[‹éR0Aa fvÛã
(¯Óÿ ø47ã§ü*Oø,–‹àøE¿´?áh|?×¼1ö¿·y_ÙT)¬ı£nÆó³ı‘älÊcí÷/cÿ W´Q_ÀQ_­ÿ ğf‹<+áÏø*ÿ Š4ø›O°»×şêö­íêE&£tº“tÖğ+f”[Û\LQ2Â8%|mF#ú¢Šş è¢Šş¿à×ùAGÀÏû™¿õ'Õ«ïú+ñÇş_ñg…lÿ à›<yâm>-oQøái§èÒ^¢İ]ZÛhú¬wÄN÷Š'ºµGuQ®aA‘süÈÑEıÁ~ñUŸ…io^xgP‹DÔu	XiúÌ–N¶·WVÉ«Éqo¤lybK«WtRYæÀ	?½ÔQ^ÿ bÿ ”Y~Òßöoş2ÿ ÓåTQ^Áÿ öø¥àOß·×Àÿ µßì¿x?ã†u¿êe–²XZj¶ÓÜMåÂ¯$›"Ûj+3c
	 W÷9EüQÁi¿iøk¿ø*¿Ço–Ú·‡õ>ïâŞ• êŸÎ°Ô4­0.™aw¢I_:ÒÎ	ZDmÒ3 TeQóW×ÿ ğ@ïÚ_ágì‰ÿ zø%ñÓã^­ıá‹O]éZ¨óÁ:ö›w¦Gwq,òGVĞËy³HÍòCŒ
³Ú+Ïÿ j¯Ú_ágìoû8øÓö£ø×«}“ÃğıÆ«©ìæºò×÷v–ş|‘Æ÷3Êc‚Ù×ÌšhĞ¸¯áŠ(¯¯ÿ àß´¿ÂÏÙş
õğKã§Æ½[û;Ã »Òµ=Qç‚tÿ í=6ïLîâYä8­¡–ò9f‘›ä†9*ÿ g´WÈğ[_ø)×ÂÏø&'ì;â¿ë?ÿ ±>#x§Ãú–•ğwM±³‚òşë]kb°İ¥´ù­­%’ç’PbUÚ„I$ĞÃ/ñ…EWîwüßÿ Sø5ğ/Qñ?üSã·ˆôÿ Kãÿ E®ü0Õî 1Ç¨ë3[Åis¦OrÒìIeK[#hZ‰$™ZYm¡èº³üYâÏ
øÂºŸ¼uâm?EÑ4]>kıgYÕ¯RÚÖÂÖ2Kq4²‘Dˆ¬ììBª©$€+ùÿ ƒˆà¯?ğõŸÛ$Â­Ö¾Óğoá·Ú4ï…Şvƒö+›Ï=-şß©M¼´Íö‰­ÓÊWòö[Ãoºfi÷üEWôıÿ Tÿ Ê,¼}ÿ gªÿ éC¯×êçş,üRğ'Àï…&ø×ñK]şËğÇƒü?{­øSû,³ı’ÂÒâo.y$ÙnÛQY›PIº
ãÿ h_‚ı¥>xãösñÖ¡¨Zh?ğ~§á½fëI•#º†ÖúÖKY^‘P’±RÈêU‡ø3¢Š+÷;şÿ F|3ı˜¾xGöÿ ‚„éÚ…‡…<)§ŞÛxkã7Wº¤–vª|ë=6öÅ#švŠ0f·†krV(ÖÊ³*$—û_ÿ bÿ ‚YÒK?gÿ ü<šÿ %Wæü×ş¸øàOx¯öKÿ ‚mêŸğœøŸ^ğş¥£jß4­^êÆÃÂ—/)µ2i²F‰&¡r"ËÜ¥¼lÖ“G-Ø2D¿ÎQE~ïÁÿ àïøU¾ğßìÑÿ Cğ÷ˆ<A™²ÂËãf“/ÛoÅ±–á:µ¡KŸ&&¸yo¡y.%X¢ÚyšIäıñgüOÿ 9ğç…u?èÿ µÎ¡¯İØió\ZèZOÃ}}.µ-´-ue,„S,±Æ†÷EËÄø.?üwñ—ş
·§'ìùğƒÂ:‡Ã‚öZƒÏw¡Kª	uÉÃ5¥Æ¤Ñ¨H¢DXd[icKÒ<×;vƒóBŠ(¢¿Sÿ à•Ÿğuwíqÿ ğø5¦şÍşiÿ |	áÍ>KoG©ø†]3YÒ#Ì"A}å\$¶0¢L±ÀğJˆ“¬0G úÅŸğ|§Š¯<+©Ùøş	©§éÚÜº|É£j·Å§½µµº(DRÍo—Ï¾Öh–h™Ô!;‡äíåÿ ı®?à¤o~3~Õ5jWÔ..4/Eu*hŞa5¶›fÎÉkÈ V#2JaY&ye-#x}QE}¿ğ·şCÿ ‚Úüğ%Ã¯	şŞş »Óôï7ì÷)ğş“®_¿™+ÊŞmö¥i=Ôøg!|É[b…EÚˆª<Cö½ÿ ‚”~Ş·¦£=çíqûUxÃÆ–“j_¯‡¯u#mu¹¶â2ÜGgm(‰KÅ
3ef%¤rŞEQ_Õoüéğ“Qøqÿ ~Æ7Ñê/üPÖõë3{momik¥“Es+Éı5Ái’ŞA •D-Åq?ê}xüÛÄÚ‚ÿ à›´Œt{m>k½'à‹/maÕ´›{ûW’-éÕfµºH.b%@heG×*êÊH>áE tQEQEQEQEQEQEQEQEQ_×ïüãÿ ((øÿ s7ş¤úµ}ÿ _(Át¾#ø«á_üûö‰ñ?ƒ¾ê.»ºø_¨èóiZaq$º‚>ëP;"ùVv÷2ŞÉ•åÚ¾çs"}_EµÀ¿øeÿ ÚŸâ_ìÑÿ 	Oöçü+¿ˆÏ†?¶şÃöoí°_Mkö'|W™åoÙ½öîÆæÆOŸÑEQEQEQEQEQEQEQEQ_ÚïüKÀğ­à‘ÿ ³—‡á5ñ¿öŸƒú«öïj_j¹‹íÖ‰}öD}«¶Úßíf·º·‚òÛ7§ëóÃş§ñg…|9ÿ 9ø»£ø‡ÄÚ}…Ş¿¨xnÃBµ½½H¤Ôn—_°ºkxˆ3J-í®&(™a¾6£úE_ğV/ùJoí-ÿ gã/ı>^W€QEQEQEQEQEQEQEQEQ_Ügü'Â~*ğüoö|ğ/¼3¨hºŞ‹ğ?ÂvÎ«Y=µÕ…Ô:=¬rÛÍ€<R£«## ÊÊA Š÷
üÿ ƒÏ|M¨è?ğJév6Ú{Å­|pÒ,¯÷I·¹’8×LÕ®A$±³ÚË¾hJHcib,bšTÖú+øÒÿ ƒ‡~
xWàü“ö€ğ/ƒµBæÒÿ Æx’i59QäZÍ¶¯t€¢ ò–âúTŒXF¨vøÂŠ(¢Š(¢Š(¢Š(¢Š(¢Š(¢Š(¢Š(¢Šşÿ (¯Æø=£Ä^;¶ı~xOOøuö¯^ü`[½_Å¿Úñ'ömüUòYØı”2o´Åq}/œ¤,_ÙûX:cözŠş@¿àèïùN¿Ç?û–õÒkà
(¢Š(¢Š(¢Š(¢Š(¢Š(¢Š(¢Š(¢Šè>ü-ñßÇŠ~ø)ğ·BşÔñ?Œ<Ae¢xsLûTP}®şît‚Ş2fHãß,ˆ»•W9b &¿½Ê+ğşœÿ ›]ÿ ¹Ûÿ pûıE*_ğx_ÁO
ü+ÿ ‚ÀÉã¯j„×¾è$×c½•8.¢{­!R ¨¥bû>•nä9vóS¸)TOË
(¢Š(¢Š(¢Š(¢Š(¢Š(¢Š(¢Š(¢Š÷ÿ ø$ïü¥7öiÿ ³€ğoşŸ,ëû}¢¿ø>ã_…uß¿ ¿g;=?P]oÂ¾Ö¼I¨]IY-u[«k[tƒ—2«è×EÃ"¨Y!*ÎY‚EÔWóÿ «ÊS|ÿ fÿ ¥éó\¯È(¢Š(¢Š(¢Š(¢Š(¢Š(¢Š(¢Š(¢Šúÿ şğ/ş#ş%û=øşŸììÿ ˆøŸíaûG™ı‰ºÏÙöïLyß`ò7äù~nı¯·cg´Wóÿ «ÊS|ÿ fÿ ¥éó\¯éúŠş|àùOéÖ¾*ıš|cÎ nïôÿ ÙOšµÃÚ¬p>èÑÚ´†e&æ@óF‹$ª°¬Œë!?h¢Š(¢Š(¢Š(¢Š(¢Š(¢Š(¢Š(¢Š+ïÿ ø5ÇşS¯ğ3şæoıF5jş¿h¯äKş§ñgŠ¼Gÿ Æø»£ø‡ÄÚ…ı¦§ønÃBµ½½ycÓ­[@°ºkx‰Än.n&(˜S$ò¾7;ıvÑ_‚?ğ|§„üUyá_Ù§ÇV~Ô%Ñ4íCÅ¶†³“µ­­ÕÊi[ÛÉ(YRÖéÑ†u¶˜¨"6ÇóãEQEQEQEQEQEQEQEQ_­ÿ ğf„ü+â?ø*ÿ Š5øgO¿»Ğ>ê÷úÕí’K&tÚ“j×3a”ÛÜÜB]0Æ9åLívú¢¿/ø9â.ñGşkñóÄÚ\¾ Óô‡_í;¬Í§é6VösÍæ[g>[8š,ùsÇÉ,1ÿ c´Wãü¿á?
ŞÁ6şxêóÃ:|ºŞñÂÒÃOÖd²Fºµµ¹ÑõY.-ã”é¯kjîŠB»[BXãù‘¢Š(¢Š(¢Š(¢Š(¢Š(¢Š(¢Š(¢Š(¯Ùïø2_áoõÛëâ¿Æ½?Bó<1áÿ ƒí¢júŸÚ¢g¿Ô5[ìáòË	Ì‹L¾mÊ¥WÈÃ.¿¥ê+ø‚ÿ ‚±ÊSioû8éòò¿°ßø&O‹<Uãßø&ßìùã¯x›PÖµ½kà„ïõgV½{›«û©´{Y%¸šY	yewfvv%™˜’I5îùÿ «Ê,¼ÿ g¥é\¯æŠ(¢Š(¢Š(¢Š(¢Š(¢Š(¢Š(¢Š(¢¿¿àÆ?ùº/û’÷?_¿ÔWğGñgâ—ş8üSñ7Æ¿ŠZïö§‰üaâİoÄzŸÙbƒíw÷s¼÷ypªGùdvÚŠª¹Â€ ı¦ÿ Á<M§x³ş	Aû6êš]¶¡Q|ğÅ“.§¤ÜYHd¶Ó-íäaÄhíx˜Ç09£),LñHßG×Àğtwü £ãŸıË?ú“é5üQEQEQEQEQEQEQEQEWôÿ Güğ®…ûüdı£,õAµ¿|PƒÃz…¬’¡µ×JÓ¡º·xÔ q+>³t³²•ª…X¿íusÿ ~)xàwÂÏ|kø¥®ÿ eøcÁş½ÖüG©ı–YşÉaiÏq7—
¼’lŠ7m¨¬ÍŒ($_Á_¿ğkü £àgıÌßú“êÕ÷ı|ÿ GÊ
>9ÿ Ü³ÿ ©>“_ÈQEQEQEQEQEQEQEQEOßğeOü¢ËÇßöp¯ş˜ô:ı~¯ ÿ ‚±Ê,¿ioû7ÿ éò¿ˆ*ÿÙimport numpy as np
a = 1.580
b = 1
c = a - b
print c
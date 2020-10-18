# -*- coding:utf-8 -*-
import cv2
import numpy as np
import time
from glob import  glob
# lower_green = np.array([35,43,46])
# upper_green = np.array([77,255,255])
lower_green = np.array([0,20,100])
upper_green = np.array([15,40,150])
#

lower_cheng = np.array([11,43,46])
upper_cheng = np.array([25,255,255])
lower_lan = np.array([78,43,46])
upper_lan = np.array([124,255,255])
red = {'low':np.array([156,43,46]),'upper':np.array([180,255,255])}
green = {'low':np.array([35,43,46]),'upper':np.array([77,255,255])}
cheng = {'low':np.array([11,150,160]),'upper':np.array([25,255,230])}
blue = {'low':np.array([78,130,46]),'upper':np.array([124,255,255])}
dict = {'red':red,'green':green,'cheng':cheng,'blue':blue}
def recongise(img,color):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,color["low"],color['upper'])
    mask = cv2.erode(mask,kernel=(2,2),iterations=2)
    mask = cv2.dilate(mask,kernel=(2,2),iterations=2)
    #轮廓检测
    cnts= cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    try:
        #找出最大轮廓
        c = max(cnts,key=cv2.contourArea)
        #确定轮廓的外接圆
        ((x,y),radius) = cv2.minEnclosingCircle(c)
        # cv2.drawContours(img,c,-1,(0,0,255),3)
        M = cv2.moments(c)
        #轮廓中心坐标
        center = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
        cv2.circle(img,(int(x),int(y)),int(radius),(0,255,255),2)
        cv2.circle(img,center,5,(0,0,255),-1)
        # print(c)
        print(key,center)
        # print('time：',round(time.time() - start,2))
        # ((x,y),radius) = cv2.minEnclosingCircle()
    except:
        # print('Threr is no light has been found !!')
        pass
    # cv2.imshow("img",mask)
    cv2.imshow('light',img)
    cv2.waitKey(1)

if __name__ == '__main__':
    cap = cv2.VideoCapture('./test5.avi')
    ret,frame = cap.read()
    n=0
    while ret:
        ret,frame = cap.read()
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imwrite("./image/"+str(n)+".jpg",frame)
        cv2.imshow("img",frame)
        cv2.waitKey(1)
        hsv = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
        n = n+1
        for key in dict:
            color =  dict[key]
            recongise(hsv,color)
    # img = cv2.imread("./test1.jpg")
    # print(img)
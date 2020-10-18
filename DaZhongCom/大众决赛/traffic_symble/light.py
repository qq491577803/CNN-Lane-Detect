
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import time
from glob import  glob
# lower_red = np.array([156,43,46])
# upper_red = np.array([180,255,255])

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
    # name = 'None'
    # location = 'None'
    name = []
    location = []

    for key in dict:
        color =  dict[key]
        mask = cv2.inRange(hsv,color["low"],color['upper'])
        mask = cv2.erode(mask,kernel=(4,4),iterations=4)
        mask = cv2.dilate(mask,kernel=(4,4),iterations=4)
        #轮廓检测
        cnts= cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        if cnts == []:
            pass
        # binary, cnts,hier= cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # try:
        else:
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
            # name = key
            # location = center
            name.append(key)
            location.append(center)
            # ((x,y),radius) = cv2.minEnclosingCircle()
            cv2.imshow('img',img)
            # cv2.imshow('img',mask)
            cv2.waitKey(0)
            # except:
            #     # print('Threr is no light has been found !!')
            #     pass
    return name,location

if __name__ == '__main__':
    fpts =  glob('./pic/*.jpg')
    for fn in fpts:
        print(fn)
        img = cv2.imread(fn)
        img = cv2.resize(img, dsize=(640, 360))
        start = time.time()
        name,location = recongise(img)
        print('Ttime:',time.time() - start)
        print(name)
        print(location)

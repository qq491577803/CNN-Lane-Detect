# -*- coding:utf-8 -*-
import cv2
import numpy as np
import time
from glob import  glob
#lower_green = np.array([35,43,46])
#upper_green = np.array([77,255,255])
lower_green = np.array([35,43,46])
upper_green = np.array([77,255,255])





red = {'low':np.array([-10,43,46]),'upper':np.array([10,255,255])}
green = {'low':np.array([35,43,46]),'upper':np.array([77,255,255])}
cheng = {'low':np.array([11,150,160]),'upper':np.array([25,255,230])}
blue = {'low':np.array([40,43,46]),'upper':np.array([255,255,255])}
dict = {'red':red,'blue':blue}
##dict = {'blue':blue}


def recongise(BGR):
    BGR = cv2.resize(BGR,(160,90))
    hsv = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)    
    res = []
    for key in dict:
        color = dict[key]
        mask = cv2.inRange(hsv,color["low"],color['upper'])
##        mask = cv2.erode(mask,kernel=(2,2),iterations=2)
##        mask = cv2.dilate(mask,kernel=(2,2),iterations=2)
        cv2.imshow('lightmask',mask)
        cv2.waitKey(1)
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
##            s = np.pi*(radius*radius)
            print 'radius',radius
            if radius<10 or radius >50:
                continue
            res.append(key)             
            # print('time：',round(time.time() - start,2))
            # ((x,y),radius) = cv2.minEnclosingCircle()
        except:
            # print('Threr is no light has been found !!')
            print 'NoLine has been detected0 !!'
            pass
    if len(res)>0:
        print '111111111111111111'
##        cv2.imshow('light',BGR)
##        cv2.waitKey(1)
        print 'rs'
##        if res[0] == 'green':
##            GreenLight = 1
##        else:
##            GreenLight = 0
        if res[0] =='red':
            RedLight =1
        else:
            RedLight = 0
        if res[0] == 'blue':
            HireSign = 1
        else:
            HireSign = 0
    else:
        RedLight = 0        
        HireSign = 0
        cv2.imshow('Nolight',BGR)
        cv2.waitKey(1)
        print "No light has been detected !!!"
        
    ParkSign = 0
    GreenLight = 0
    print "lightflag:",GreenLight,RedLight,ParkSign,HireSign
    return GreenLight,RedLight,ParkSign,HireSign
if __name__ == '__main__':

    fpts =  glob('./green.jpg')
    for fn in fpts:
        img = cv2.imread(fn)
        img = cv2.resize(img, dsize=(640, 360))
        start = time.time()
        res = recongise(img)
        print'light_time :',time.time()-start
        print(res)
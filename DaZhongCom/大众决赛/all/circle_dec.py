# -*- coding: utf-8 -*-
import cv2
import numpy as np
pth = './pic/all.jpg'

# img =  cv2.imread("./lan.jpg")

def circle_dec(img):
    img = cv2.resize(img, dsize=(320, 180))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150)
    cv2.imshow("canny", canny)
    cv2.waitKey(1)
    circles = cv2.HoughCircles(canny, method=cv2.cv.CV_HOUGH_GRADIENT, dp=1, minDist=100, param1=150, param2=30, minRadius=1,maxRadius=200)
    try:
        for circle in circles[0]:
            # 圆心的坐标
            x = int(circle[0])
            y = int(circle[1])
            # 半径
            r = int(circle[2])
            print('Circle Radius:', r,x,y)            
            Green = 1
        cv2.circle(img, (int(x), int(y)), int(r), (0, 0, 255), -1)
        cv2.imshow("res", img)
        cv2.waitKey(1)
    except:
            print('NoCircle ...')
            Green = 0
    print '......................circle'


if __name__ == '__main__':
    img =  cv2.imread('./circle.jpg')
####    img = cv2.circle(img, (50, 50), 5, (0, 0, 255), -1)
####    cv2.circle(img,(int(50),int(50)),int(20),(0,255,255),2)
##    cv2.imshow('11',img)
##    cv2.waitKey(0)
    
    circle_dec(img)



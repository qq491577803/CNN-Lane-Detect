# -*-coding: utf-8 -*-
import cv2
import numpy as np
import time
import datetime
backup_flag = 0
last_flag = 0
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
    global backup_flag
    global last_flag
    print ("mask _shape________________________:",mask.shape)
##    mask = cv2.resize(mask,dsize=(16,9))#
    # mask = cv2.resize(mask,(160,90))
    # cv2.imshow('dec_mask',mask)
    mask = (mask/255)
    line_point = mask[10]
    line_point1 = mask[70]
    list = []
    list1 = []
    axis = []
    axis1 = []
    for i in range(len(line_point) - 1):
        par = abs((int(line_point[i + 1]) - int(line_point[i])))
        list.append(par)
        if par ==1:
            axis.append(i)
    counter = sum(list)
    for j in range(len(line_point1) - 1):
        par1 = abs((int(line_point1[j + 1]) - int(line_point1[j])))
        list1.append(par1)
        if par1 == 1:
            axis1.append(j)
    counter1 = sum(list1)
    print('The counter is ：',counter,counter1,(line_point1.shape))
    if 6 < counter<20 or 6<counter1<20:
        flag = 1 
        print('Decrease lane has been detect ...')
    else:
        flag = 0
        print ('NO dec_lane')
    backup_flag = flag
    if flag - last_flag == 1:
        flag = 1
    else:
        flag = 0
    last_flag = backup_flag
    print(axis)
    print(axis1)
    print(len(axis),len(axis1),',,')
    if len(axis)>=len(axis1):

        y_axis = axis[2]
    else:
        y_axis = axis1[2]
    print(y_axis,'//')
    mask[:,y_axis:360]=0
    cv2.imshow("mask",mask)
    cv2.waitKey(0)

    return flag,
def sobel(warped,thresh = (30,180)):
    gray = cv2.cvtColor(warped,cv2.COLOR_RGB2GRAY)
    sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,1))
    scal_sobel = np.uint8(255*sobel/np.max(sobel))
    mask = np.zeros_like(scal_sobel)
    mask[(scal_sobel >=thresh[0]) & (scal_sobel <=thresh[1])] = 1
    mask = mask *255
    
    return mask
def hsv(warped):
    lower_black = np.array([0,0,0])
    upper_black = np.array([180,255,70])
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_black, upper_black)
    return mask

if __name__ == '__main__':
    img = cv2.imread('./321.jpg')
    img = cv2.resize(img, dsize=(320, 180))    
    frame = warp(img)
##    mask0 = sobel(frame)
    mask0 = hsv(frame)
    # cv2.imshow('=mask0',mask0)
    # cv2.waitKey(1)
    st = time.time()
    num = dec_lane(mask0)
    print('total:',time.time()-st)
    #rightangle(mask0)

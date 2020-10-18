import cv2
from glob import glob
import os
import numpy as np
def th():
    pth = './training/*_lanes.png'
    fns = glob(pth)
    for fn in fns:
        img = cv2.imread(fn)
        print(img)
        img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = np.where(img<100,0,255)
        # cv2.imshow('img',np.array(img,dtype=np.uint8))
        # cv2.waitKey(0)
        img = img / 255
        cv2.imshow("img",img*255)
        cv2.waitKey(1)
def resize(pth):
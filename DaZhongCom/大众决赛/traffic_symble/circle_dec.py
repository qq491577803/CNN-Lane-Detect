import cv2
import numpy as np
pth = './pic/all.jpg'
from glob import  glob
# img =  cv2.imread("./lan.jpg")

def circle_dec(img):
    img = cv2.resize(img, dsize=(320, 180))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150)
    cv2.imshow("canny", canny)
    circles = cv2.HoughCircles(canny, method=cv2.HOUGH_GRADIENT, dp=1, minDist=90, param1=100, param2=30, minRadius=6,maxRadius=150)
    print("leng_circle:",len(circles[0]))
    try:
        for circle in circles[0]:
            # 圆心的坐标
            x = int(circle[0])
            y = int(circle[1])
            # 半径
            r = int(circle[2])
            print('Circle Radius:', r)
            img = cv2.circle(img, (x, y), r, (0, 0, 255), -1)
            cv2.imshow("res", img)
            cv2.waitKey(0)
    except:
            print('NoCircle ...')
# cap = cv2.VideoCapture(0)
# ret,frame = cap.read()
# while ret:
#     ret,img = cap.read()
#     cv2.imshow("orign",img)
#     circle_dec(img)

path = glob('./8-25/*.jpg')
for p in path:
    img = cv2.imread(p)
    circle_dec(img)

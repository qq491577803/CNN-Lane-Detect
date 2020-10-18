
import cv2

cap =  cv2.VideoCapture(0)
ret,frame = cap.read()
while 1:
    ret,frame = cap.read()
    cv2.imshow("test",frame)
    cv2.waitKey(1)
    
import cv2
from cnn_lane import

def perspective():
    src = np.float32([[33,116], [55,25], [248,25], [282,116]])
    dst = np.float32([[33, 180], [33, 0], [282, 0], [282,180]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M,Minv
def warp(img):
    M,Minv = perspective()
    img_size = (320, 180)
    warped = cv2.warpPerspective(img, M, img_size)
    #cv2.imshow('warp',warped)
##    cv2.waitKey(1)
    return warped
def hsv(warped):
    lower_black = np.array([0,0,0])
    upper_black = np.array([180,255,75])
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_black, upper_black)
    return mask



cap = cv2.VideoCapture('./test9.avi')
ret,frame = cap.read()
n = 0
while ret:
    n = n +1
    print(n)
    ret,frame = cap.read()




cap.release()
import numpy as np
import cv2
import tkinter.filedialog

def perspective():
    src = np.float32([[32,101], [59,13], [252,13], [288,101]])
    dst = np.float32([[32, 180], [32, 0], [288, 0], [288,180]])

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
def inverse_warp(img):
    M,Minv = perspective()
    unwarp_img = cv2.warpPerspective(img,Minv,(880,246))
    return unwarp_img
if __name__ == '__main__':
    fname = tkinter.filedialog.askopenfilename()
    img = cv2.imread(fname)
    img = cv2.resize(img,dsize=(880,246))
    warp_img = warp(img)



    # cv2.imwrite(fname + 'warp.png',warp_img)
    # cv2.imshow("warp",warp_img)
    # cv2.waitKey(0)
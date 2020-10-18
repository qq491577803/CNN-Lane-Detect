import numpy as np
import cv2
import tkinter.filedialog

def perspective():
    src = np.float32([[102, 243], [410, 23],[483, 23],[836, 243]])
    dst = np.float32(([[514, 5951], [514, 0], [1284,0], [1284, 5951]]))
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    return M,Minv
def warp(img):
    M,Minv = perspective()
    warp_img = cv2.warpPerspective(img,M,(1762,5951))
    return warp_img
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
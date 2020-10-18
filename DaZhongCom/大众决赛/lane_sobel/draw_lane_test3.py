# -*- coding:utf - 8 -*-
import cv2
import numpy as np
from lane_detection import Polyfit
import time
from time import sleep

def perspective():
    src = np.float32([[40,138], [72,27], [240,27], [280,138]])
    dst = np.float32([[40, 180], [40, 0], [280, 0], [280,180]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M,Minv
def warp(img):
    M,Minv = perspective()
    img_size = (320, 180)
    warped = cv2.warpPerspective(img, M, img_size)
    cv2.imwrite("./waped2.jpg",warped)
    cv2.imshow("warp",warped)
    cv2.waitKey(0)
    print('warp_shape:',warped.shape)
    return warped
def draw(img, warped, left_fit, right_fit):
    M,Minv = perspective()
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # cv2.imshow("color",color_warp)
    plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
    right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]
    middle_fit_x = (left_fit_x+right_fit_x)/2
    y_axis = warped.shape[0]
    left = left_fit[0]*y_axis**2 + left_fit[1]*y_axis + left_fit[2]
    right = right_fit[0]*y_axis**2 + right_fit[1]*y_axis + right_fit[2]
    lane_center = (left+right)/2
    offset = (warped.shape[1]/2 - lane_center)*(35/277)
    dy = 2*left_fit[0]*147 + left_fit[1]
    ddy = 2*left_fit[0]
    R = ((1+dy**2)**(3/2))/ddy*35/277

    pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
    pts_middle = np.array([np.flipud(np.transpose(np.vstack([middle_fit_x, plot_y])))])
    PTS =np.hstack(pts_middle)
    pts = np.hstack((pts_left, pts_right))
    try:
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0),)
        cv2.polylines(color_warp,np.int_([PTS]),isClosed=False,color=(0,0,255),thickness=5)
        cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(255, 150,0),thickness=8)
        # cv2.imwrite("./lane_persp.jpg")
        # cv2.imshow('colormap',color_warp)
    except:
        pass
    # cv2.imwrite("./draw_lane.jpg",color_warp)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    try:
        cv2.putText(result, str('Radius:') + str(round(R, 2)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 5), 1)
        cv2.putText(result, str('offset:') + str(round(offset, 2)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 5), 1)
        pass
    except:
        pass
    # cv2.imshow("img",result)
    # cv2.waitKey(0)
    return result,offset,R
def hsv(warp):
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 70])
    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_black, upper_black)
    cv2.imshow('mask', mask*255)
    cv2.imwrite("soble.jpg", mask * 255)
    return mask
def abs_threshold(warped, orient='x', thresh=(30,180)):
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    mask = np.zeros_like(scaled_sobel)
    mask[(scaled_sobel >= thresh[0]) &(scaled_sobel <= thresh[1])] = 1
    cv2.imshow('abs',mask*255)
    cv2.imwrite("soble.jpg",mask*255)
    return mask



def auto_lane(img):
    polyfit = Polyfit()
    img = cv2.resize(img,dsize=(320,180))
    cv2.imwrite('./resize.jpg',img)
    warped = warp(img)
    mask = abs_threshold(warped)
    # mask = hsv(warped)
    # frame = combined_threshold(frame)
    left_fit, right_fit, vars = polyfit.poly_fit_skip(mask)
    if left_fit =='None'  or right_fit == 'None':
        result,offset,Radius = None,None,None
        cv2.imshow('re', cv2.resize(img, dsize=(853, 480)))
        cv2.waitKey(0)
        return result,offset,Radius
    else:
        result,offset,Radius = draw(img, mask, left_fit, right_fit)
        cv2.imshow('re',cv2.resize(result,dsize=(853,480)))
        cv2.waitKey(0)
    cv2.imwrite("./res.jpg",result)
    return result,offset,Radius

if __name__ == "__main__":
    # polyfit = Polyfit()
    # original_video = './video.mp4'
    # cap = cv2.VideoCapture(original_video)
    # n = 0
    # while (1):
    #     n = n+1
    #     ret,img = cap.read()
    #     start = time.time()
    #     result, offset, Radius = auto_lane(img)
    #     print('Time per px :',time.time() - start)
    #     print(offset,Radius)
    img = cv2.imread('./2.jpg')
    # cv2.imwrite('./3.jpg',cv2.resize(img,dsize=(320,180)))
    auto_lane(img)

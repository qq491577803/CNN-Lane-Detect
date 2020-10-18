import cv2
import numpy as np
from lane_detection import Polyfit
from lane_detection import combined_threshold
# from moviepy.video.io.VideoFileClip import VideoFileClip
import time
from time import sleep

def perspective():
    src = np.float32([[189, 317], [278, 255], [373, 255], [480, 319]])
    dst = np.float32([[100, 200], [100, 0], [391, 0], [391,200]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M,Minv

def warp(img):
    M,Minv = perspective()
    img_size = (491, 200)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped

def draw(img, warped, left_fit, right_fit):
    print(img.shape,warped.shape)
    M,Minv = perspective()
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
    right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]
    middle_fit_x = (left_fit_x+right_fit_x)/2

    dy = 2*left_fit[0]*710 + left_fit[1]
    ddy = 2*left_fit[0]
    R = ((1+dy**2)**(3/2))/ddy*2.87/584

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
    cv2.imwrite("./draw_lane.jpg",color_warp)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    try:
        cv2.putText(result, str('Radius of curvature:') + str(round(R, 2)), (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 5), 1)
        pass
    except:
        pass
    # cv2.imshow("img",result)
    # cv2.waitKey(0)
    return result

def draw_lane(img_arr):
    polyfit = Polyfit()
    combined_output = combined_threshold(img_arr)
    warped = warp(combined_output)
    left_fit, right_fit, vars = polyfit.poly_fit_skip(warped)
    # print(left_fit,right_fit)
    result = draw(img_arr, warped, left_fit, right_fit)
    return result,left_fit,right_fit


if __name__ == "__main__":

    polyfit = Polyfit()
    original_video = './video.mp4'
    cap = cv2.VideoCapture(original_video)
    n = 0
    while (1):
        n = n+1
        ret,img = cap.read()
        start = time.time()
        # cv2.imshow("iimg",frame)
        img = cv2.resize(img,dsize=(640,320))
        frame = warp(img)
        frame = combined_threshold(frame)
        left_fit, right_fit, vars = polyfit.poly_fit_skip(frame)
        result = draw(img, frame, left_fit, right_fit)
        end = time.time()
        print('time',end -start)
        cv2.imshow('re', result)
        cv2.waitKey(1)

    #1280  720



    # polyfit = Polyfit()
    # img = cv2.imread('./lane1.jpg')
    # frame = warp(img)
    # print(frame.shape)
    # frame = combined_threshold(frame)
    # print('222',frame.shape)
    # left_fit, right_fit, vars = polyfit.poly_fit_skip(frame)
    # result = draw(img, frame, left_fit, right_fit)
    # cv2.imshow('re',result)
    # cv2.waitKey(0)
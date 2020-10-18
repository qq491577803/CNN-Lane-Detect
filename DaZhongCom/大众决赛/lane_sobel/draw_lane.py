import cv2
import numpy as np
from lane_detection import Polyfit
from lane_detection import combined_threshold
# from moviepy.video.io.VideoFileClip import VideoFileClip
import time

def perspective():
    src = np.float32([[370, 720], [954, 720], [552, 574], [745, 575]])
    dst = np.float32([[280, 720], [1000, 720], [280, 0], [1000, 0]])
    # src = np.float32([[32, 659], [1156, 659], [424, 422], [801, 422]])
    # dst = np.float32([[200, 720], [1100, 720], [200, 0], [1100, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M,Minv

def warp(img):
    M,Minv = perspective()
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size)
    return warped

def unwarp(img):
    M,Minv = perspective()
    img_size = (img.shape[1], img.shape[0])
    unwarped = cv2.warpPerspective(img, M, img_size)
    return unwarped

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

    original_video = './video.mp4'
    cap = cv2.VideoCapture(original_video)
    n = 0
    while (1):
        n = n+1
        ret,frame = cap.read()
        # cv2.imshow("iimg",frame)
        result, left_fit, right_fit = draw_lane(frame)
        # cv2.imshow("img",img_warp)
        cv2.imshow('lane',cv2.resize(result,dsize=(640,360)))
        cv2.waitKey(1)

    #


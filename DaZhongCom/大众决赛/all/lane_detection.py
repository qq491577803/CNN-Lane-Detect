# -*- coding:utf - 8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
class Polyfit:
    def __init__(self):
        self.left_fit = 'None'
        self.right_fit = None
        self.left_x = None
        self.right_x = None
        self.left_y = None
        self.right_y = None
        self.ym_per_pix = 30 / 720  # y方向上每个像素对应的实际长度
        self.xm_per_pix = 3.7 / 700  # x方向。。。
        self.margin = 80

    def poly_fit_skip(self, img):
        """
        拟合车道线
        """
        # 检测是否存在车道线
        if self.left_fit is None or self.right_fit is None:
            return self.poly_fit_slide(img)        #
        nonzero = img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        margin = self.margin
        left_lane_inds = ((nonzero_x > (self.left_fit[0] * (nonzero_y**2) + self.left_fit[1] * nonzero_y + self.left_fit[2] - margin)) & (
            nonzero_x < (self.left_fit[0] * (nonzero_y**2) + self.left_fit[1] * nonzero_y + self.left_fit[2] + margin)))
        right_lane_inds = ((nonzero_x > (self.right_fit[0] * (nonzero_y**2) + self.right_fit[1] * nonzero_y + self.right_fit[2] - margin)) & (
            nonzero_x < (self.right_fit[0] * (nonzero_y**2) + self.right_fit[1] * nonzero_y + self.right_fit[2] + margin)))
        # 提取左右两车道的像素坐标
        self.left_x = nonzero_x[left_lane_inds]
        self.left_y = nonzero_y[left_lane_inds]
        self.right_x = nonzero_x[right_lane_inds]
        self.right_y = nonzero_y[right_lane_inds]
        # 拟合
        self.left_fit = np.polyfit(self.left_y, self.left_x, 2)
        self.right_fit = np.polyfit(self.right_y, self.right_x, 2)
        vars = {}
        vars['left_fit'] = self.left_fit
        vars['right_fit'] = self.right_fit
        vars['nonzero_x'] = nonzero_x
        vars['nonzero_y'] = nonzero_y
        vars['left_lane_inds'] = left_lane_inds
        vars['right_lane_inds'] = right_lane_inds

        return self.left_fit, self.right_fit, vars

    def poly_fit_slide(self, img):
        """
        沿图像下半部分的所有列取直方图，实现滑动窗口以查找并追踪直到图像顶部的行将二阶多项式拟合到每一行
        """
        #图像下半部分的直方图
        histogram = np.sum(img[img.shape[0]//2:, :], axis=0)
        out_img = np.dstack((img, img, img)) * 255
        midpoint = np.int(histogram.shape[0] / 2)
        left_x_base = np.argmax(histogram[:midpoint])
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint
        #滑动窗口的个数
        num_windows = 5
        window_height = np.int(img.shape[0] / num_windows)
        nonzero = img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        left_x_current = left_x_base
        right_x_current = right_x_base
        margin = self.margin
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []
        for window in range(num_windows):
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = left_x_current - margin
            win_xleft_high = left_x_current + margin
            win_xright_low = right_x_current - margin
            win_xright_high = right_x_current + margin
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)
            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (
                nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (
                nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > minpix:
                left_x_current = np.int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > minpix:
                right_x_current = np.int(np.mean(nonzero_x[good_right_inds]))
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        self.left_x = nonzero_x[left_lane_inds]
        self.left_y = nonzero_y[left_lane_inds]
        self.right_x = nonzero_x[right_lane_inds]
        self.right_y = nonzero_y[right_lane_inds]        
        if self.left_x.size ==0 and self.left_y.size ==0 and self.right_y.size == 0 and self.right_x.size == 0 or self.left_x.size <=1:
            self.left_fit, self.right_fit,vars = 'None','None',None
            print('There is no lane has been detected ```')
            return self.left_fit, self.right_fit, vars
        elif self.left_x.size >1 and self.right_x.size <=1 and self.left_y.size >1 or self.right_y.size <=1:
            print 'right lane detect has been failed !'
            self.right_fit,vars = 'None',None
            self.left_fit = np.polyfit(self.left_y, self.left_x, 2)            
            return self.left_fit, self.right_fit, vars
        
            
        else:
            print self.left_y.size,self.right_y.size
            self.left_fit = np.polyfit(self.left_y, self.left_x, 2)
            self.right_fit = np.polyfit(self.right_y, self.right_x, 2)
            vars = {}
            vars['left_fit'] = self.left_fit
            vars['right_fit'] = self.right_fit
            vars['nonzero_x'] = nonzero_x
            vars['nonzero_y'] = nonzero_y
            vars['left_lane_inds'] = left_lane_inds
            vars['right_lane_inds'] = right_lane_inds
            vars['out_img'] = out_img
            vars['margin'] = margin

            return self.left_fit, self.right_fit, vars


if __name__ =="__main__":
    pass
    # # img0 = cv2.imread("./test_image.jpg")
    # # img = warp(img0)
    #
    # #sobel
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    # abs_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # cv2.imwrite("sobel_x.jpg",abs_sobel)
    # cv2.imshow("sobel_x",abs_sobel)
    # abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # abs_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # cv2.imshow("sobel_y",abs_sobel)
    # cv2.imwrite("sobel_y.jpg",abs_sobel)
    # #abs
    # abs_thresh = abs_threshold(img, orient='x', thresh=(20, 100))
    # cv2.imshow("abs",abs_thresh*255)
    # cv2.imwrite('./abs.jpg',abs_thresh*255)
    # cv2.waitKey(0)
    # #mag
    # mag_thresh = mag_threshold(img, sobel_kernel=3, mag_thresh=(50, 100))
    # cv2.imshow("mag",mag_thresh*255)
    # cv2.imwrite('./mag.jpg',mag_thresh*255)
    # cv2.waitKey(0)
    # #dir
    # dir_thresh = dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.2))
    # cv2.imwrite("dir.jpg",dir_thresh)
    # cv2.imshow("dir", dir_thresh * 255)
    # cv2.waitKey(0)
    # #hls
    # hls_threshold = hls_threshold(img, thresh=(100, 255))
    # cv2.imshow("hls", hls_threshold * 255)
    # cv2.imwrite("hls.jpg",hls_threshold)
    # cv2.waitKey(0)
    #
    # #hsv
    # hsv_thresh = hsv_threshold(img, thresh=([20, 100, 100], [35, 255, 255], [0, 0, 230], [180, 25, 255]))
    # cv2.imshow("hsv", hsv_thresh * 255)
    # cv2.imwrite("hsv.jpg",abs_sobel)
    # cv2.waitKey(0)
    #
    # #combine
    # combine = combined_threshold(img)
    # cv2.imshow("combie",combine*255)
    # cv2.imwrite("./combinee.jpg",combine*255)
    # cv2.waitKey(0)

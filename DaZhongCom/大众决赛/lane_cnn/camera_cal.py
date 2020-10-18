import numpy as np
import cv2
from glob import glob
import os

def calibrate_chessboard():
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    calibration_fnames = glob('./chess_board/*.jpg')
    calibration_images = []
    objpoints = []#世界坐标
    imgpoints = []#实际坐标

    NUM = 0
    for fname in calibration_fnames:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        calibration_images.append(gray)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret == True:
            NUM = NUM +1
            print(True)
            print(NUM)
            objpoints.append(objp)
            imgpoints.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,calibration_images[0].shape[::-1], None, None)
    calibration = [mtx, dist]
    np.save('./chess_board/calibration.npy',calibration)
    return calibration
def undistort(img_arr,calibration=None):
    calibration = np.load('./chess_board/calibration.npy')
    return cv2.undistort(img_arr,calibration[0],calibration[1],None,calibration[0])
if __name__ == "__main__":
    calibration = calibrate_chessboard()
    # fnames = glob('./chess_board/IMG_20180522_19*.jpg')
    # for fname in fnames:
    #     image = cv2.imread(fname)
    #
    #     cv2.imshow("original",image)
    #     img = undistort(img_arr=image)
    #     cv2.imshow("undistort",img)
    #     cv2.waitKey(0)
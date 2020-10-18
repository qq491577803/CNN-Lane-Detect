import cv2
import numpy as np
from lane_detection import combined_threshold
from lane_detection import Polyfit
def perspective_1(img):
    '''    此透视变换是用来计算车辆距离的，和 Perspective不一样
    :param img: 传入的是图像矩阵
    :return: 透视变换后的图像矩阵
    '''
    # src = np.float32([[609, 440],[673, 440],[289, 670],[1032, 670]])
    # dst = np.float32([[2232, 0],[2976, 0],[2232, 6032],[2976, 6032]])
    src = np.float32([[370, 720], [954, 720], [552, 574], [745, 575]])
    dst = np.float32([[1752,7216],[2336, 7216],[1752, 0],[2336, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    # warped = cv2.warpPerspective(img, M, (5208,6032))
    warped = cv2.warpPerspective(img, M, (4088, 7216))
    return warped
def direction_calculate(x,y):
    '''
    计算目标车量的角度，当角度大于0，说明在右方，并和中心线的夹角为 alpha
    :param x:目标车辆的坐标
    :param y:
    :return:flag是判断是否在俯视图中检测到目标车辆的坐标，alpha 是目标车辆的方位
    '''

    zero_arr=np.zeros(shape=[1024,1024])
    if x >1023:
        x =1022
    elif y>1022:
        y = 1022
    zero_arr[y,x] = 255
    zero_img = cv2.resize(zero_arr,(1080,720))
    zero_img = perspective_1(zero_img)

    raw,colum= zero_img.shape
    position = np.argmax(zero_img)
    if position == 0:
        print("The point is not visionalbe in perspective image !")
        flag = 0
        n =None
        m = None
        alpha =None
        return flag,alpha,n,m,raw,colum
    else:
        n,m = divmod(position,colum)
        b = ((colum-colum/2)**2 + (raw-raw)**2)**0.5
        c = ((m-colum/2)**2 + (n-raw)**2)**0.5
        a = ((m-colum)**2 + (n-raw)**2)**0.5
        cosA = (c**2+b**2-a**2)/(2*b*c)
        A = np.arccos(cosA)*180/np.pi
        print(A)
        alpha = np.round((90 - A),1)
        if alpha >0:
            print("The objection is on the right of center {A}{B}".format(A=abs(alpha),B="°"))
        elif alpha< 0 :
            print("The objection is on the left of center {A}{B}".format(A=abs(alpha), B="°"))
        else:
            print("The objection is in the front {A}{B}".format(A=abs(alpha), B="°"))
        flag = 1
        return flag,alpha,n,m,raw,colum
def distance_calculate(n,m,raw,colum):
    '''
    次函数用来计算目标车辆距离
    :param n,m: 函数输入的是目标车辆在透视变换后的图像中的坐标
    :param raw,colum: 图像行列数
    :return: 距离，单位米
    '''
    # per_pixel_meter_x = 3.7/744#根据标定求得
    # per_pixel_meter_y = 3.7/744
    per_pixel_meter_x = 2.87/584#根据标定求得
    per_pixel_meter_y = 2.87/584

    x_pixel = abs(m-colum/2)
    y_pixel = abs(n-raw)
    print("x:",x_pixel,'y:',y_pixel)
    x_meter = x_pixel * per_pixel_meter_x
    y_meter = y_pixel * per_pixel_meter_y
    distance = (x_meter**2+y_meter**2)**0.5
    return distance
def off_center(img_arr,left_fit, right_fit):
    """
    :param img_arr: 矫正后的图片矩阵
    :param left_fit: 车道左侧拟合系数
    :param right_fit: 车道右侧拟合系数
    :return: 偏离车道的距离，单位m，返回的是一个字符串
    """
    plot_y = img_arr.shape[0]
    left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
    right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]
    lane_center = (left_fit_x+right_fit_x)/2
    # off_center = (img_arr.shape[1]/2 - lane_center)*(3.77/744)
    off_center = (img_arr.shape[1]/2 - lane_center)*(2.87/584)
    off_center = round(off_center,3)
    if off_center<0:
        print("Offset to right:",abs(off_center),str("m"))
        center_messages = "Offset to right: : " + str(abs(off_center))+str("m")
        return  center_messages
    if off_center>0:
        print("Offset to left:",abs(off_center),str("m"))
        center_messages = "Offset to left:" + str(abs(off_center))+str("m")
        return  center_messages
    if off_center == 0:
        print("On the center:",abs(off_center),str("m"))
        center_messages = "On the center:" + str(abs(off_center))+str("m")
        return  center_messages
if __name__ == "__main__":
    # x,y =
    ######计算距离
    x,y=197,894
    flag,alpha,n,m,raw,colum = direction_calculate(x,y)
    if  flag == 1 :
        distance = distance_calculate(n,m,raw,colum)
        print(flag,alpha,distance)
    else:
        pass
    #####计算偏离中心的距离
    img_arr =cv2.imread("./input/test6.jpg")
    polyfit = Polyfit()

    combined_output = combined_threshold(img_arr)
    warped = perspective_1(combined_output)
    left_fit, right_fit, vars = polyfit.poly_fit_skip(warped)
    offset = off_center(img_arr,left_fit,right_fit)

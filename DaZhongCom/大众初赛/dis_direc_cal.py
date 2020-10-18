import cv2
import numpy as np

def perspective_1(img):
    '''    此透视变换是用来计算车辆距离的，和 Perspective不一样
    :param img: 传入的是图像矩阵
    :return: 透视变换后的图像矩阵
    '''
    # src = np.float32([[609, 440],[673, 440],[289, 670],[1032, 670]])
    # dst = np.float32([[2232, 0],[2976, 0],[2232, 6032],[2976, 6032]])
    # src = np.float32([[81, 716],[1180, 716],[673, 379],[725, 379]])
    # dst = np.float32([[2847, 0],[3796, 0],[2847, 8389],[3796, 8389]])
    src = np.float32([[1269, 3113],[3385, 3113],[1982, 1727],[2443, 1727]])
    dst = np.float32([[4240,9540],[6360, 9540],[4240, 0],[6360, 0]])


    M = cv2.getPerspectiveTransform(src, dst)
    # warped = cv2.warpPerspective(img, M, (5208,6032))
    warped = cv2.warpPerspective(img, M, (10600, 9540))
    return warped


def direction_calculate(x,y):
    '''
    计算目标车量的角度，当角度大于0，说明在右方，并和中心线的夹角为 alpha
    :param x:目标车辆的坐标
    :param y:
    :return:flag是判断是否在俯视图中检测到目标车辆的坐标，alpha 是目标车辆的方位
    '''

    zero_arr=np.zeros(shape=[1024,1024])

    zero_arr[y,x] = 255
    zero_img = cv2.resize(zero_arr,(1080,720))
    zero_img = perspective_1(zero_img)

    cv2.imwrite("./zero.jpg",zero_img)
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
    per_pixel_meter_x = 2.87/949#根据标定求得
    per_pixel_meter_y = 2.87/949

    x_pixel = abs(m-colum/2)
    y_pixel = abs(n-raw)
    print("x:",x_pixel,'y:',y_pixel)
    x_meter = x_pixel * per_pixel_meter_x
    y_meter = y_pixel * per_pixel_meter_y
    distance = (x_meter**2+y_meter**2)**0.5
    return distance

if __name__ =='__main__':
    img = cv2.imread('./dist_dire_cal/4_0.jpg')
    img = perspective_1(img)
    cv2.imwrite('./dist_dire_cal/4.jpg',img)
    cv2.imshow('img',img)
    cv2.waitKey(0)
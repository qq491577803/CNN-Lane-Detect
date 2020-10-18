import cv2
import numpy as np
def perspective():
    src = np.float32([[44,118], [82,14], [228,14], [270,118]])
    dst = np.float32([[47, 180], [47, 0], [273, 0], [273,180]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M,Minv
def warp(img):
    M,Minv = perspective()
    img_size = (320, 180)
    warped = cv2.warpPerspective(img, M, img_size)
    cv2.imwrite('resize.jpg',warped)
    cv2.imshow('warp',warped)
    cv2.waitKey(1)
    return warped
def hsv(warped):
    lower_black = np.array([0,0,0])
    upper_black = np.array([180,255,70])
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_black, upper_black)
    return mask
def corner_det(mask,img):
    edgs = mask[0:54,0:180]
    cv2.imshow('edgs',edgs)
    cv2.waitKey(1)
    lines = cv2.HoughLinesP(edgs,2,np.pi/180,15,minLineLength=30,maxLineGap=10)
    try:
        lines1 = lines[:,0,:]#提取为二维
        sp_line = []
        sz_line = []
        sp_k = []
        sz_k = []
        for x1,y1,x2,y2 in lines1[:]:
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
            k = (y2-y1)/(x2-x1)
            print('kkkkk',k)
            if abs(k) < 0.05:
                length = np.sqrt((y2-y1)**2+(x2-x1)**2)
                if  85<length<180:
                    print('------------------------------',length)
                    sp_line.append((x1,y1,x2,y2))
                    sp_k.append(k)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            # else:
            #     sz_line.append((x1,y1,x2,y2))
            #     sz_k.append(k)
            #     cv2.line(img,(x1, y1), (x2, y2), (255, 0, 0), 1)
        if len(sp_k) == 0:
            print("None corner...")
        if len(sp_k) != 0:
            # if len(sz_k) != 0:
            #     sp_k = abs(np.mean(sp_k))
            #     sz_k = abs(np.mean(sz_k))
            #     if sp_k < 0.2 and sz_k > 500:
            #         print('zhi jiao wan !!!')
            #         cv2.putText(img, str('corner:'), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # else:
            #     cv2.putText(img, str('corner:'), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #     print("zhi jiao wan ")
            cv2.putText(img, str('corner:'), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            print("zhi jiao wan ")
            print('K1','k2',sp_k, sz_k)
            cv2.imshow("img", img)
            cv2.waitKey(1)
    except:
        print("None line ...")

    return 0

if __name__ == "__main__":
    import time
    cap =  cv2.VideoCapture("./test6.avi")
    ret,frame = cap.read()
    while ret:
        ret,frame = cap.read()
        img = cv2.resize(frame, dsize=(320, 180))
        cv2.imshow("frame",img)
        warped = warp(img)
        hsved = hsv(warped)
        strat = time.time()
        corner_det(hsved,img)
        print(time.time()-strat)
        cv2.imshow("hsved",hsved)
        cv2.waitKey(1)




        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # edgs = cv2.Canny(gray, 50, 150, apertureSize=3)
        # cv2.imshow("canny", edgs)
        # corner_det(edgs,img)
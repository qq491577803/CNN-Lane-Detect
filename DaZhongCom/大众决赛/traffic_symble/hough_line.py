import cv2
import numpy as np


def corner_det(edgs,img):
    edgs = edgs[0:100,0:300]
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
            if abs(k) < 0.2:
                sp_line.append((x1,y1,x2,y2))
                sp_k.append(k)
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            else:
                sz_line.append((x1,y1,x2,y2))
                sz_k.append(k)
                cv2.line(img,(x1, y1), (x2, y2), (255, 0, 0), 1)
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
            print(sp_k, sz_k)
            cv2.imshow("img", img)
            cv2.waitKey(1)
    except:
        print("None line ...")

    return 0
if __name__ == "__main__":
    img = cv2.imread('./390.jpg')
    img = cv2.resize(img, dsize=(720, 360))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edgs = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imshow("canny", edgs)
    corner_det(edgs,img)
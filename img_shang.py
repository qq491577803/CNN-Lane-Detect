#！usr/bin/env python
#encoding:utf-8
'''
__Author__:Lsz
Function:
'''
import cv2
import numpy as np

def main(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    src = np.array(gray,dtype=np.float16)
    print("src shape : ",src.shape)
    m,n = 40,40
    ncol = int(src.shape[0] / m + 0.5)












    nraw = int(src.shape[1] / n + 0.5)

    for i in range(ncol):
        for j in range(nraw):
            x_begin = i * m
            x_end = min(x_begin + m - 1,src.shape[0]-1)
            y_begin = j * n
            y_end = min(y_begin + n-1,src.shape[1]-1)
            # print(i,j,x_end,y_end)
            roi =  src[x_begin:x_end,y_begin:y_end]
            # print("roi shape: ",roi.shape)
            entropy = cal_entropy(roi)
            img = cv2.putText(gray, str(round(entropy,2)), (x_begin, y_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("res",gray)
    cv2.waitKey(0)

def cal_entropy(array):
    res = [0 for i in range(256)]
    entropy = 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            res[int(array[i,j])] += 1
    res = [x/(array.shape[0] * array.shape[1]) for x in res]
    res = [x*np.log2(x) for x in res  if x !=0]
    for x in res:
        entropy = entropy - x
    return entropy






if __name__ == '__main__':
    path = r"C:\Users\Administrator\Desktop\2019.2.3-2019.2.9\img\img.jpg"
    main(path)
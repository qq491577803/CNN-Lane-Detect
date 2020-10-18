import cv2
import numpy as np
def light_recongize(img_arr):
    img = cv2.resize(img_arr,(90,140))
    img = img[10:140, 10:80]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = 255 - th2
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=np.ones((1, 1), np.uint8))
    img = cv2.resize(img, (90, 140))
    light_name = {'pos1': 'Red Light', 'pos2': 'Yellow Light', 'pos3': 'Green Light'}
    img1 = img[15:48, 15:75]
    pos1 = len(np.where(img1 < 10)[0])
    img2 = img[48:90, 15:75]
    pos2 = len(np.where(img2 < 10)[0])
    img3 = img[90:125, 15:75]
    pos3 = len(np.where(img3 < 10)[0])
    # print(pos1,pos2,pos3)
    if pos1 < 300 and pos2 < 300 and pos3 < 300:
        message = '-- --'
    elif pos1 > pos2 >= pos3 or pos1 > pos3 >= pos2:
        message = light_name['pos1']
    elif pos2 > pos1 >= pos3 or pos2 > pos3 >= pos1:
        message = light_name['pos2']
    elif pos3 > pos2 >= pos1 or pos3 > pos1 >= pos2:
        message = light_name['pos3']
    else:
        message = '--'
    return message

if __name__ =='__main__':
    fpath = "./LightExmple/"
    N = 1
    for i in range(523):
        path = fpath + str(N) + '.jpg'
        img = cv2.imread(path)
        message =  light_recongize(img)
        print(message)
        N=N+1
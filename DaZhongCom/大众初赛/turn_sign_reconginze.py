import cv2
from glob import glob
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from skimage import feature as ft
import time
def perspective_1(img):
    src = np.float32([[81, 720],[1092, 720],[550, 469],[767, 469]])
    dst = np.float32([[0, 1075],[1011, 1075],[0, 0],[1011,0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (1011, 1075))
    return warped



turn_name={
    0:'None',1:'turn right',2:'turn left',3:'go straight'
}
labels = np.array([[3,3,0,0,0,0,0,1,3,3,3,3,3,3,3,2,1,1,2,2,2,0,0,0]])
labels = np.reshape(labels,[24,1])

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]
y_one_hot = convert_to_one_hot(labels,5)
y_one_hot= np.delete(y_one_hot,[0],axis=1)
fnames = glob("./test_image/turn/*jpg")
hog_arr = []
for fname in fnames:
    img = cv2.imread(fname)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img =cv2.GaussianBlur(img,(3,3),sigmaX=1,sigmaY=1)
    # hogmap, hogimage = ft.hog(img, orientations=9, pixels_per_cell=(20, 20), cells_per_block=(2, 2),block_norm='L1',
    #                              transform_sqrt=True, feature_vector=True, visualise=True)
    hog = np.reshape(img,[1,280*312])
    hog_arr.append(hog)
hog_arr = np.array(hog_arr)
hog_arr = np.reshape(hog_arr,[24,280*312])

#训练模型并保存
def train_model():
    clf = svm.SVC(C=1, gamma=0.1, kernel='rbf', decision_function_shape="ovr")
    print("Training ...")
    clf.fit(hog_arr, labels.ravel())
    print("Saving model ...")
    joblib.dump(clf, "./model/turn.model")
    print("Calculating the Train Accuracy ...")
    Accuracy = accuracy_score(labels, clf.predict(hog_arr))
    print("Train Accuracy is :", Accuracy)
    print("Train model has been completed ...")
    acc = clf.predict(hog_arr)
    print(acc)

def predict(img_arr):
    turn_name = {
        0: 'None', 1: 'turn right', 2: 'turn left', 3: 'go straight'
    }
    img = cv2.resize(img_arr, (280, 312))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.reshape(img, [1, 280*312])
    clf = joblib.load("./model/turn.model")
    pre_num = int(clf.predict(img))
    message = turn_name[pre_num]
    return message
if __name__ == "__main__":
    # fnames = glob('./test_image/turn/*.jpg')
    # for fname in fnames:
    #     img =  cv2.imread(fname)
    #     message = predict(img)
    #     print(message)
    img =cv2.imread('./input/turn.jpg')
    # cv2.imshow("img", img)
    # img = perspective_1(img)
    # cv2.imwrite("./turn_pers.jpg",img)
    # cv2.imshow("pers",img)
    # img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # hogmap, hogimage = ft.hog(img, orientations=9, pixels_per_cell=(20, 20), cells_per_block=(2, 2),block_norm='L1',
    #                              transform_sqrt=True, feature_vector=True, visualise=True)
    # cv2.imwrite("./turn_hog.png",hogimage)
    # cv2.imshow("hog",hogimage)
    # cv2.waitKey(0)
    begin = time.time()
    res = predict(img)
    end = time.time()
    print(end-begin)
    print(res)

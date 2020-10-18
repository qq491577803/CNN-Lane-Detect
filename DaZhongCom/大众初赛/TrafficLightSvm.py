import cv2
from glob import glob
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
light_name={1:"Red",2:"None",3:"Green",4:'Yellow'}

#读取标签，放入到label矩阵中
def  getLabels(n,path):
    print("Reading Labels ...")
    Lb_arr=np.zeros([n,1],int)
    with open(path) as f:
        lines=f.readlines()[0:n]
        arr=-1
        for line in lines:
            arr=arr+1
            list=line.strip("\n")
            Lb_arr[arr]=list
        f.close()
    return Lb_arr
#读取图片矩阵放入到img_arr中
def getImg_arr(n,path):
    img_arr = []
    #此处遍历训练文件不能使用os.listdir更不能使用glob，文件顺序和标签不对应了
    num = 1
    for i in range(n):
        fp = path+str(num)+str('.jpg')
        img = cv2.imread(fp)
        img = cv2.resize(img,(90,140))
        img = np.reshape(img,[1,140*90*3])
        img_arr.append(img)
        num = num+1
    img_arr = np.array(img_arr)
    img_arr =np.reshape(img_arr,[n,37800])
    return img_arr
#训练模型并保存
def train_model(img_arr,labels,test_img=None,test_label=None):
    clf = svm.SVC(C=1, gamma=0.1, kernel='rbf', decision_function_shape="ovr")
    print("Training ...")
    clf.fit(img_arr, labels.ravel())
    print("Saving model ...")
    joblib.dump(clf, "./model/light.model")
    print("Calculating the Train Accuracy ...")
    TrainAccuracy = accuracy_score(labels, clf.predict(img_arr))
    print("Train Accuracy is :", TrainAccuracy)
    if test_img is not None:
        TestAccuracy = accuracy_score(test_label, clf.predict(test_img))
        print('Test Accuracy is :', TestAccuracy)
    print("Train model has been completed ...")
#模型预测
def predict(img_arr):
    light_name = {1: "Red", 2: "None", 3: "Green", 4: 'Yellow'}
    img = cv2.resize(img_arr, (90, 140))
    img = np.reshape(img, [1, 140 * 90 * 3])
    clf = joblib.load("./model/light.model")
    pre_num = int(clf.predict(img))
    name = light_name[pre_num]
    print(name)
    return name
if __name__ == "__main__":
    #测试样本读取
    # #训练样本读取
    # label = getLabels(523,'./LightExmple/label/label.txt')
    # img_arr = getImg_arr(523,'./LightExmple/')
    # print(label.shape)
    # print(img_arr.shape)
    # permutation = np.random.permutation(523)
    # label = label[permutation]
    # img_arr = img_arr[permutation]
    # #训练
    # train_model(img_arr, label)
    # fs = glob('./LightExmple/test_sample/*.jpg')
    fs =glob('./configs/*.jpg')
    for f in fs:
        img_a  = cv2.imread(f)
        print(f)
        predict(img_a)



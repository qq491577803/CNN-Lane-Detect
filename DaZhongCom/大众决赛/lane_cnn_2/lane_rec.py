import cv2
from glob import glob
import numpy as np
import tensorflow as tf
import time
from draw_line import draw
from lane_detection import Polyfit

def perspective():
    src = np.float32([[32,101], [59,13], [252,13], [288,101]])
    dst = np.float32([[32, 180], [32, 0], [288, 0], [288,180]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M,Minv

def warp(img):
    M,Minv = perspective()
    img_size = (320, 180)
    warped = cv2.warpPerspective(img, M, img_size)
    #cv2.imshow('warp',warped)
##    cv2.waitKey(1)
    return warped


def read_data():
    x = glob('x/*.jpg')
    lanes = glob('y/*.jpg')
    x.sort()
    lanes.sort()
    X = []
    for pth in x:
        print(pth)
        arr = cv2.imread(pth)
        X.append(arr)
    X = np.stack(X)
    Y = []
    for pth in lanes:
        arr = cv2.imread(pth)
        arr = cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        arr = np.where(arr < 100, 0, 255)
        arr = arr/255
        Y.append(arr)
    Y = np.reshape(np.stack(Y),newshape=(-1,180,320,1))
    data = {'x':X,'y':Y}
    return data


def data_read():
    pth_x= './x/'
    pth_y= './y/'
    X = []
    Y = []
    for i in range(25):
        path = pth_x + str(i+1) + '.jpg'
        print(path)
        img = cv2.imread(path)
        img = cv2.resize(img,dsize=(107,60))
        X.append(img)
    X = np.stack(X)
    print('X.shape : ',X.shape)
    for i in range(25):
        path = pth_y + str(i+1) + '.jpg'
        print(path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img,dsize=(107,60))
        img = np.where(img < 200, 0, 255)
        img = img /255
        Y.append(img)
    Y = np.reshape(np.stack(Y), newshape=(-1, 60, 107, 1))
    data = {'x': X, 'y': Y}
    return data




def train_data_read():
    pth_x= './videoimage/'
    pth_y= './video_hsv/'
    print(pth_x,pth_y)
    X = []
    Y = []
    for i in range(50):
        path = pth_x + str(i+1) + '.jpg'
        # print(path)
        img = cv2.imread(path)
        img = cv2.resize(img,dsize=(107,60))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        X.append(img)
    print(len(X),'-=-=-=-=-=-=-')
    X = np.stack(X)
    X = np.resize(X,new_shape=(50,60,107,1))
    print('X.shape : ',X.shape)
    for j in range(50):
        path = pth_y + str(j+1) + '.jpg'
        # print(path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img,dsize=(107,60))
        img = np.where(img < 200, 0, 255)
        img = img /255
        Y.append(img)
    Y = np.reshape(np.stack(Y), newshape=(-1, 60, 107, 1))
    print('Y.shape',Y.shape)
    data = {'x': X, 'y': Y}
    return data

def wight_varible(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def train(epochs=4000):
    data = train_data_read()
    print(data['x'].shape)
    # with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 60, 107, 1],name = 'x_input')
    y = tf.placeholder(tf.float32, shape=[None, 60, 107, 1],name = 'y_input')
    prob = tf.placeholder(tf.float32,name='prob')
    # '''��һ��'''
    with tf.name_scope('layers'):
        with tf.name_scope('layer1'):
            with tf.name_scope('W1'):
                W_conv1 = wight_varible([2, 2, 1, 15])
            with tf.name_scope('b1'):
                b_conv1 = bias_variable([15])
            with tf.name_scope('h1'):
                h_conv1 = (conv2d(x, W_conv1) + b_conv1)  # (?,123,440,20)
            # with tf.name_scope('BN1'):
            #     h_conv1 = tf.contrib.layers.batch_norm(inputs=h_conv1,decay=0.9,is_training=True,updates_collections=None)
            with tf.name_scope('relu1'):
                h_conv1 = tf.nn.relu(h_conv1)
            with tf.name_scope('drop_out1'):
                h_conv1 = tf.nn.dropout(h_conv1, prob)
        '''�ڶ���'''
        with tf.name_scope('layers2'):
            with tf.name_scope('W2'):
                W_conv2 = wight_varible([3, 3, 15, 10])
            with tf.name_scope('b2'):
                b_conv2 = bias_variable([10])
            with tf.name_scope('h2'):
                h_conv2 = (conv2d(h_conv1, W_conv2) + b_conv2)  # (?,123,440,30)
            # with tf.name_scope('BN2'):
            #     h_conv2 = tf.contrib.layers.batch_norm(inputs=h_conv2, decay=0.9, is_training=True, updates_collections=None)
            with tf.name_scope('relu2'):
                h_conv2 = tf.nn.relu(h_conv2)
            with tf.name_scope('drop_out2'):
                h_conv2 = tf.nn.dropout(h_conv2, prob)
        '''������'''
        with tf.name_scope('layers3'):
            with tf.name_scope('W3'):
                W_conv3 = wight_varible([2, 2, 10, 5])
            with tf.name_scope('b3'):
                b_conv3 = bias_variable([5])
            with tf.name_scope('h3'):
                h_conv3 = (conv2d(h_conv2, W_conv3) + b_conv3)  # (?,123,440,30)
            with tf.name_scope('BN3'):
                h_conv3 = tf.contrib.layers.batch_norm(inputs=h_conv3, decay=0.9, is_training=True, updates_collections=None)
            with tf.name_scope('relu3'):
                h_conv3 = tf.nn.relu(h_conv3)
            with tf.name_scope('drop_out3'):
                h_conv3 = tf.nn.dropout(h_conv3, prob)
        '''���߲�'''
        with tf.name_scope('layer7'):
            with tf.name_scope('W7'):
                W_conv7 = wight_varible([3, 3, 5, 1])
            with tf.name_scope('b7'):
                b_conv7 = bias_variable([1])
            with tf.name_scope('relu7'):
                h_conv7 = tf.nn.relu(conv2d(h_conv3, W_conv7) + b_conv7)  # (?,123,440,1)
    # h_conv7 = tanh_zero_to_one(conv2d(h_conv6, W_conv7) + b_conv7)

    tf.add_to_collection('pred_network', h_conv7)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y, logits=h_conv7, pos_weight=50))
        # loss = tf.reduce_mean(tf.square(y - h_conv7))
        # tf.summary.scalar('loss',loss)

    with tf.name_scope('optimister'):
        optimis = tf.train.AdamOptimizer().minimize(loss)
    print("Training ...")
    saver = tf.train.Saver()
    tf.add_to_collection('pred_network',h_conv7)
    loss_courve = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # writer = tf.summary.FileWriter('./logs/',sess.graph)
        with open('./model_test/loss.txt', 'w') as fn:
            for epoch in range(epochs):
                sess.run(optimis, feed_dict={x: data['x'], y: data['y'],prob:0.7})
                loss_val = sess.run(loss, feed_dict={x: data['x'], y: data['y'], prob: 1})
                print('Epoch', epoch, ' Loss:',loss_val )
                message = 'Epoch,' + str(epoch) + ',' + 'Loss,' + str(loss_val)
                fn.write(message)
                fn.write('\n')
                if (epoch+1) % 10 == 0:
                    saver.save(sess, "./model_test/lane-conv--490-", global_step=epoch)
            fn.close()
def test_img_process(pth):
    x = glob('training/*_x.png')
    x.sort()
    X = []
    for pth in x:
        arr = cv2.imread(pth)
        arr = arr[420:666, 200:1080]
        arr = cv2.resize(arr, (440, 123))
        X.append(arr)
    X = np.stack(X)
    return X




def predict(video_path):
    polyfit = Polyfit()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loader = tf.train.import_meta_graph('./model_test/lane-conv--490--299.meta')
        loader.restore(sess, './model_test/lane-conv--490--299')
        y = tf.get_collection('pred_network')[0]
        graph = tf.get_default_graph()
        input_x = graph.get_operation_by_name('x_input').outputs[0]
        keep_prob = graph.get_operation_by_name('prob').outputs[0]
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        n =0
        while ret:
            ret, frame = cap.read()
            # cv2.imshow("img",frame)
            img_arr = cv2.resize(frame, dsize=(107, 60))
            img_arr = cv2.cvtColor(img_arr,cv2.COLOR_BGR2GRAY)
            imgarr = np.reshape(img_arr, newshape=[1, 60, 107, 1])
            predict = sess.run(y, feed_dict={input_x: imgarr, keep_prob: 1.0})[0]
            predict = np.uint8(np.where(predict==0,0,255))
            predict = cv2.resize(predict,dsize=(320,180))
            cv2.imwrite("./predict/"+str(n)+'.jpg',predict)
            n = n+1
            cv2.imshow('pre',predict)
            cv2.waitKey(1)
def cnn_lane(sess,imgarr):
    imgarr = np.reshape(imgarr, newshape=[1, 60, 107, 3])
    predict = sess.run(y, feed_dict={input_x: imgarr, keep_prob: 1.0})[0]
    return predict

if __name__ == "__main__" :
    predict('./test9.avi')
    #
    # data = train_data_read()
    # print(data['x'].shape,data['y'].shape)
    train(3000)
# # Single image process
#     filename = selectfile()
#     img_arr  = cv2.imread(filename)
#     start = time.time()
#     result = prediction(img_arr)
#     print('time:',time.time()-start)
#     cv2.imshow('result',result)
#     cv2.waitKey(0)    # Video process
#     VideoProcee(input_path='./test5.avi',output_path='./lane_detect.avi',timeF=1,fps=30)
#     train()
#     prediction()
    # data_read()
    sess = tf.Session()
    polyfit = Polyfit()
    sess.run(tf.global_variables_initializer())
    loader = tf.train.import_meta_graph('./model_test/lane-conv---99.meta')
    loader.restore(sess, './model_test/lane-conv---99')
    y = tf.get_collection('pred_network')[0]
    graph = tf.get_default_graph()
    input_x = graph.get_operation_by_name('x_input').outputs[0]
    keep_prob = graph.get_operation_by_name('prob').outputs[0]
    cap = cv2.VideoCapture('./test5.avi')
    ret, frame = cap.read()
    while ret:
        s =time.time()
        ret, frame = cap.read()
        imgarr = cv2.resize(frame, dsize=(107, 60))
        predict = cnn_lane(sess,imgarr)
        predict = np.uint8(np.where(predict==0,0,255))
        predict = cv2.resize(predict,dsize=(320,180))
        warped = warp(predict)
        left_fit, right_fit, vars = polyfit.poly_fit_slide(warped)
        result, offset, Radius, k_error = draw(cv2.resize(frame,dsize=(320,180)),warped, left_fit, right_fit)
        print('Total;',time.time() - s)
        cv2.imshow('warped',warped)
        cv2.imshow('res',result)
        cv2.waitKey(1)
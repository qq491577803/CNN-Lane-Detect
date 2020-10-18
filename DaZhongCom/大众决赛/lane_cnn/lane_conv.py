# -*- coding:utf-8--
import cv2
from glob import glob
import numpy as np
import tensorflow as tf
import time
def read_data():
    x = glob('training/*_x.png')
    lanes = glob('training/*_lanes.png')
    x.sort()
    lanes.sort()
    X = []
    for pth in x:
        arr = cv2.imread(pth)
        arr = cv2.resize(arr,dsize=(160,90))
        X.append(arr)
    X = np.stack(X)
    print(X.shape)
    Y = []
    for pth in lanes:
        arr = cv2.imread(pth)
        arr = cv2.resize(arr, dsize=(160,90))
        arr = cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        arr = np.where(arr < 10, 0, 255)
        # cv2.imshow("img",np.array(arr,dtype=np.uint8))
        # cv2.waitKey(0)
        arr = arr / 255
        # cv2.imshow("img",np.array(arr*255,dtype=np.uint8))
        Y.append(arr)
    Y = np.reshape(np.stack(Y),newshape=(-1,90,160,1))
    data = {'x':X,'y':Y}
    return data
def wight_varible(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def train(epochs=4000):
    data = read_data()
    x = tf.placeholder(tf.float32, shape=[None, 90, 160, 3],name = 'x_input')
    y = tf.placeholder(tf.float32, shape=[None, 90, 160, 1],name = 'y_input')
    prob = tf.placeholder(tf.float32, name='prob')

    '''第一层'''
    with tf.name_scope('layers'):
        with tf.name_scope('layer1'):
            with tf.name_scope('W1'):
                W_conv1 = wight_varible([3, 3, 3, 20])
            with tf.name_scope('b1'):
                b_conv1 = bias_variable([20])
            with tf.name_scope('h1'):
                h_conv1 = (conv2d(x, W_conv1) + b_conv1)  # (?,123,440,20)
            # with tf.name_scope('BN1'):
            #     h_conv1 = tf.contrib.layers.batch_norm(inputs=h_conv1,decay=0.9,is_training=True,updates_collections=None)
            with tf.name_scope('relu1'):
                h_conv1 = tf.nn.relu(h_conv1)
            with tf.name_scope('drop_out1'):
                h_conv1 = tf.nn.dropout(h_conv1, prob)
        '''第二层'''
        with tf.name_scope('layers2'):
            with tf.name_scope('W2'):
                W_conv2 = wight_varible([5, 5, 20, 10])
            with tf.name_scope('b2'):
                b_conv2 = bias_variable([10])
            with tf.name_scope('h2'):
                h_conv2 = (conv2d(h_conv1, W_conv2) + b_conv2)  # (?,123,440,30)
            # with tf.name_scope('BN2'):
                # h_conv2 = tf.contrib.layers.batch_norm(inputs=h_conv2, decay=0.9, is_training=True, updates_collections=None)
            with tf.name_scope('relu2'):
                h_conv2 = tf.nn.relu(h_conv2)
            with tf.name_scope('drop_out2'):
                h_conv2 = tf.nn.dropout(h_conv2, prob)
        '''第三层'''
        # with tf.name_scope('layers3'):
        #     with tf.name_scope('W3'):
        #         W_conv3 = wight_varible([3, 3, 10, 20])
        #     with tf.name_scope('b3'):
        #         b_conv3 = bias_variable([20])
        #     with tf.name_scope('h3'):
        #         h_conv3 = (conv2d(h_conv2, W_conv3) + b_conv3)  # (?,123,440,30)
        #     # with tf.name_scope('BN3'):
        #     #     h_conv3 = tf.contrib.layers.batch_norm(inputs=h_conv3, decay=0.9, is_training=True, updates_collections=None)
        #     with tf.name_scope('relu3'):
        #         h_conv3 = tf.nn.relu(h_conv3)
        #     with tf.name_scope('drop_out3'):
        #         h_conv3 = tf.nn.dropout(h_conv3, prob)
        '''第四层'''
        with tf.name_scope('layers3'):
            with tf.name_scope('W3'):
                W_conv4 = wight_varible([5, 5, 10, 10])
            with tf.name_scope('b3'):
                b_conv4 = bias_variable([10])
            with tf.name_scope('h3'):
                h_conv4 = (conv2d(h_conv2, W_conv4) + b_conv4)  # (?,123,440,30)
            # with tf.name_scope('BN4'):
            #     h_conv4 = tf.contrib.layers.batch_norm(inputs=h_conv4, decay=0.9, is_training=True, updates_collections=None)
            with tf.name_scope('relu3'):
                h_conv4 = tf.nn.relu(h_conv4)
            with tf.name_scope('drop_out3'):
                h_conv4 = tf.nn.dropout(h_conv4, prob)
        '''第五层'''
        # with tf.name_scope('layer5'):
        #     with tf.name_scope('W5'):
        #         W_conv5 = wight_varible([3, 3, 15, 5])
        #     with tf.name_scope('b5'):
        #         b_conv5 = bias_variable([5])
        #     with tf.name_scope('h5'):
        #         h_conv5 = (conv2d(h_conv4, W_conv5) + b_conv5)  # (?,123,440,20)
        #     # with tf.name_scope('bn'):
        #     #     h_conv5 = tf.contrib.layers.batch_norm(inputs=h_conv5, decay=0.9, is_training=True, updates_collections=None)
        #     with tf.name_scope('relu5'):
        #         h_conv5 = tf.nn.relu(h_conv5)
        #     with tf.name_scope('drop_out5'):
        #         h_conv5 = tf.nn.dropout(h_conv5, prob)
        '''第六层'''
        # with tf.name_scope('layer6'):
        #     with tf.name_scope('W6'):
        #         W_conv6 = wight_varible([3, 3, 10, 5])
        #     with tf.name_scope('b4'):
        #         b_conv6 = bias_variable([5])
        #     with tf.name_scope('h6'):
        #         h_conv6 = (conv2d(h_conv5, W_conv6) + b_conv6)  # (?,123,440,10)
        #     # with tf.name_scope('bn6'):
        #     #     h_conv6 = tf.contrib.layers.batch_norm(inputs=h_conv6, decay=0.9, is_training=True, updates_collections=None)
        #     with tf.name_scope('relu6'):
        #         h_conv6 = tf.nn.relu(h_conv6)
        #     with tf.name_scope('drop_out6'):
        #         h_conv6 = tf.nn.dropout(h_conv6, prob)
        '''第七层'''
        with tf.name_scope('layer4'):
            with tf.name_scope('W4'):
                W_conv7 = wight_varible([5, 5, 10, 1])
            with tf.name_scope('b4'):
                b_conv7 = bias_variable([1])
            with tf.name_scope('relu4'):
                h_conv7 = tf.nn.relu(conv2d(h_conv4, W_conv7) + b_conv7)  # (?,123,440,1)
    # h_conv7 = tanh_zero_to_one(conv2d(h_conv6, W_conv7) + b_conv7)

    tf.add_to_collection('pred_network', h_conv7)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y, logits=h_conv7, pos_weight=50))
        tf.summary.scalar('loss',loss)
    # loss = tf.reduce_mean(tf.square(y - h_conv7))
    with tf.name_scope('optimister'):
        optimis = tf.train.AdamOptimizer().minimize(loss)
    print("Training ...")
    saver = tf.train.Saver()
    tf.add_to_collection('pred_network',h_conv7)


    loss_courve = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./logs/',sess.graph)
        with open('./model_test/loss.txt', 'w') as fn:
            for epoch in range(epochs):
                sess.run(optimis, feed_dict={x: data['x'], y: data['y'],prob:0.7})
                summary,_ = sess.run([loss_courve,optimis],feed_dict={x: data['x'], y: data['y'], prob: 1})
                writer.add_summary(summary,epoch)

                loss_val = sess.run(loss, feed_dict={x: data['x'], y: data['y'], prob: 1})
                print('Epoch', epoch, ' Loss:',loss_val )
                message = 'Epoch,' + str(epoch) + ',' + 'Loss,' + str(loss_val)
                fn.write(message)
                fn.write('\n')
                if (epoch+1) %1 == 0:
                    # predict = sess.run(h_conv7, feed_dict={x: data['x'], y: data['y'],prob:1})
                    # img_arr = predict[1]
                    # print('epoch', epoch, ' Loss:', sess.run(loss, feed_dict={x: data['x'], y: data['y'], prob: 1}))
                    # cv2.imwrite('./observe/epoch' + str(epoch) + '.png', np.uint8(img_arr * 255))
                    saver.save(sess, "./model_test/lane-conv", global_step=epoch)
                    # print(predict[1])
                    # cv2.imshow("img",np.uint8(np.where(predict[1]>0.5,254,0)))
                    # cv2.waitKey(0)
            fn.close()

def test_img_process(pth):
    x = glob('training/*_x.png')
    x.sort()
    X = []
    for pth in x:
        arr = cv2.imread(pth)
        X.append(arr)
    X = np.stack(X)
    return X
def prediction():
    # data = read_training_data(lane_settings)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loader = tf.train.import_meta_graph('./model_test/lane-conv-1.meta')
        loader.restore(sess, './model_test/lane-conv-1')
        # loader = tf.train.import_meta_graph(r'C:\Users\Administrator\Downloads\model/lane-799.meta')
        # loader.restore(sess, r'C:\Users\Administrator\Downloads\model\lane-799')
        y = tf.get_collection('pred_network')[0]
        graph = tf.get_default_graph()
        input_x = graph.get_operation_by_name('x_input').outputs[0]
        keep_prob = graph.get_operation_by_name('prob').outputs[0]
        fpaths = glob('./training/*_x.png')

        print(fpaths)
        for fp in fpaths:
            print(fp)
            img = cv2.imread(fp)
            img = cv2.resize(img, dsize=(160,90))
            cv2.imshow("img",img)

            imgarr = np.reshape(img,newshape=[1,90,160,3])
            # cv2.imshow("img",imgarr[0])
            # cv2.waitKey(0)
            start =  time.time()
            predict = sess.run(y, feed_dict={input_x: imgarr, keep_prob: 1.0})
            print(time.time() - start)
            # cv2.imwrite(str(fp)+'.png',(predict[0]*200))
            cv2.imshow(str(fp),255 * np.uint8(predict[0]))
            cv2.waitKey(0)
            cv2.waitKey(0)

if __name__ == "__main__" :
    # train()
    prediction()


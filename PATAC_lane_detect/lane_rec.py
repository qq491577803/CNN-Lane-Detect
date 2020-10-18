import cv2
from glob import glob
import numpy as np
import tensorflow as tf
from skimage import measure
import skimage
import tkinter.filedialog
from perspective import warp,inverse_warp,perspective
import time
def selectfile():
    filename = tkinter.filedialog.askopenfilename()
    return filename
def read_data():
    x = glob('training/*_x.png')
    lanes = glob('training/*_lanes.png')
    x.sort()
    lanes.sort()
    X = []
    for pth in x:
        arr = cv2.imread(pth)
        arr = arr[420:666, 200:1080]
        arr = cv2.resize(arr,(440,123))
        X.append(arr)
    X = np.stack(X)
    Y = []
    for pth in lanes:
        arr = cv2.imread(pth)
        arr = cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        arr = arr[420:666, 200:1080]
        arr = cv2.resize(arr,(440,123))
        arr = arr / 255
        Y.append(arr)
    Y = np.reshape(np.stack(Y),newshape=(-1,123,440,1))
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
def train(epochs=4000):
    data = read_data()
    # with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 123, 440, 3],name = 'x_input')
    y = tf.placeholder(tf.float32, shape=[None, 123, 440, 1],name = 'y_input')
        # with tf.name_scope('drop_out0'):
    prob = tf.placeholder(tf.float32,name='prob')
    # '''��һ��'''
    with tf.name_scope('layers'):
        with tf.name_scope('layer1'):
            with tf.name_scope('W1'):
                W_conv1 = wight_varible([2, 2, 3, 20])
            with tf.name_scope('b1'):
                b_conv1 = bias_variable([20])
            with tf.name_scope('h1'):
                h_conv1 = (conv2d(x, W_conv1) + b_conv1)  # (?,123,440,20)
            with tf.name_scope('BN1'):
                h_conv1 = tf.contrib.layers.batch_norm(inputs=h_conv1,decay=0.9,is_training=True,updates_collections=None)
            with tf.name_scope('relu1'):
                h_conv1 = tf.nn.relu(h_conv1)
            with tf.name_scope('drop_out1'):
                h_conv1 = tf.nn.dropout(h_conv1, prob)
        '''�ڶ���'''
        with tf.name_scope('layers2'):
            with tf.name_scope('W2'):
                W_conv2 = wight_varible([3, 3, 20, 30])
            with tf.name_scope('b2'):
                b_conv2 = bias_variable([30])
            with tf.name_scope('h2'):
                h_conv2 = (conv2d(h_conv1, W_conv2) + b_conv2)  # (?,123,440,30)
            with tf.name_scope('BN2'):
                h_conv2 = tf.contrib.layers.batch_norm(inputs=h_conv2, decay=0.9, is_training=True, updates_collections=None)
            with tf.name_scope('relu2'):
                h_conv2 = tf.nn.relu(h_conv2)
            with tf.name_scope('drop_out2'):
                h_conv2 = tf.nn.dropout(h_conv2, prob)
        '''������'''
        with tf.name_scope('layers3'):
            with tf.name_scope('W3'):
                W_conv3 = wight_varible([5, 5, 30, 30])
            with tf.name_scope('b3'):
                b_conv3 = bias_variable([30])
            with tf.name_scope('h3'):
                h_conv3 = (conv2d(h_conv2, W_conv3) + b_conv3)  # (?,123,440,30)
            with tf.name_scope('BN3'):
                h_conv3 = tf.contrib.layers.batch_norm(inputs=h_conv3, decay=0.9, is_training=True, updates_collections=None)
            with tf.name_scope('relu3'):
                h_conv3 = tf.nn.relu(h_conv3)
            with tf.name_scope('drop_out3'):
                h_conv3 = tf.nn.dropout(h_conv3, prob)
        '''���Ĳ�'''
        with tf.name_scope('layers4'):
            with tf.name_scope('W4'):
                W_conv4 = wight_varible([3, 3, 30, 30])
            with tf.name_scope('b4'):
                b_conv4 = bias_variable([30])
            with tf.name_scope('h4'):
                h_conv4 = (conv2d(h_conv3, W_conv4) + b_conv4)  # (?,123,440,30)
            with tf.name_scope('BN4'):
                h_conv4 = tf.contrib.layers.batch_norm(inputs=h_conv4, decay=0.9, is_training=True, updates_collections=None)
            with tf.name_scope('relu4'):
                h_conv4 = tf.nn.relu(h_conv4)
            with tf.name_scope('drop_out4'):
                h_conv4 = tf.nn.dropout(h_conv4, prob)
        '''�����'''
        with tf.name_scope('layer5'):
            with tf.name_scope('W5'):
                W_conv5 = wight_varible([5, 5, 30, 20])
            with tf.name_scope('b5'):
                b_conv5 = bias_variable([20])
            with tf.name_scope('h5'):
                h_conv5 = (conv2d(h_conv4, W_conv5) + b_conv5)  # (?,123,440,20)
            with tf.name_scope('bn'):
                h_conv5 = tf.contrib.layers.batch_norm(inputs=h_conv5, decay=0.9, is_training=True, updates_collections=None)
            with tf.name_scope('relu5'):
                h_conv5 = tf.nn.relu(h_conv5)
            with tf.name_scope('drop_out5'):
                h_conv5 = tf.nn.dropout(h_conv5, prob)
        '''������'''
        with tf.name_scope('layer6'):
            with tf.name_scope('W6'):
                W_conv6 = wight_varible([3, 3, 20, 10])
            with tf.name_scope('b4'):
                b_conv6 = bias_variable([10])
            with tf.name_scope('h6'):
                h_conv6 = (conv2d(h_conv5, W_conv6) + b_conv6)  # (?,123,440,10)
            with tf.name_scope('bn6'):
                h_conv6 = tf.contrib.layers.batch_norm(inputs=h_conv6, decay=0.9, is_training=True, updates_collections=None)
            with tf.name_scope('relu6'):
                h_conv6 = tf.nn.relu(h_conv6)
            with tf.name_scope('drop_out6'):
                h_conv6 = tf.nn.dropout(h_conv6, prob)
        '''���߲�'''
        with tf.name_scope('layer7'):
            with tf.name_scope('W7'):
                W_conv7 = wight_varible([5, 5, 10, 1])
            with tf.name_scope('b7'):
                b_conv7 = bias_variable([1])
            with tf.name_scope('relu7'):
                h_conv7 = tf.nn.relu(conv2d(h_conv6, W_conv7) + b_conv7)  # (?,123,440,1)
    # h_conv7 = tanh_zero_to_one(conv2d(h_conv6, W_conv7) + b_conv7)

    tf.add_to_collection('pred_network', h_conv7)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y, logits=h_conv7, pos_weight=50))
        # loss = tf.reduce_mean(tf.square(y - h_conv7))
        tf.summary.scalar('loss',loss)

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
                if (epoch+1) % 1 == 0:
                    saver.save(sess, "./model_test/lane-conv1", global_step=epoch)
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
def prediction(img_arr):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loader = tf.train.import_meta_graph('./model_test/lane-conv1-0.meta')
        loader.restore(sess, './model_test/lane-conv1-0')
        y = tf.get_collection('pred_network')[0]
        graph = tf.get_default_graph()
        input_x = graph.get_operation_by_name('x_input').outputs[0]
        keep_prob = graph.get_operation_by_name('prob').outputs[0]
        arr = img_arr[420:666, 200:1080]
        cv2.imwrite('./orign_crop.jpg',arr)

        arr = cv2.resize(arr, (440, 123))
        imgarr = np.reshape(arr,newshape=[1,123,440,3])
        predict = sess.run(y, feed_dict={input_x: imgarr, keep_prob: 1.0})
        predict_img = (predict[0]*255)
        predict_img = cv2.resize(predict_img,(880,246))
        return predict_img
def output_process(img_arr):
    # img_gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    # img_threshold = cv2.threshold(img_arr, 100, maxval=255, type=cv2.THRESH_BINARY)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(img_arr, kernel1)
    eroded = cv2.erode(eroded, kernel2)
    dilated = cv2.dilate(eroded, kernel1)
    img = cv2.dilate(dilated, kernel2)
    img = np.where(img<100,0,1)
    labels = measure.label(img, connectivity=2)
    arr = skimage.measure.regionprops(labels)
    for i in arr:
        if i.area < 60:
            axis = i.coords
            for n in axis:
                img[n[0], n[1]] = 0
    # img = np.uint8(img * 255)
    return img
def find_lane_xyaxis(img):
    img = warp(np.uint8(img * 255))
    expected_x_starts = [500, 1254]  # #[520,1264]���峵��ͼ����������ʼ����
    xyaxis = [[], []]#����һ����list������������������ߵ�����
    search_range = int(248)  # #������Χ���Գ����߿�ȵ�����֮һ��������
    y_iters = int(12)  # #���峵��ͼ����������������
    lane_pixels = img.nonzero()#ȡ�������߶�ֵͼ���з�������꣬������������
    lane_pixels_y = np.array(lane_pixels[0])#ȡ����������������
    lane_pixels_x = np.array(lane_pixels[1])#ȡ�������ߺ�������
    for lane_idx in range(2):
        start_x = expected_x_starts[lane_idx]
        flag = False #����flag�������ж��Ƿ���box�д��ڳ�����
        for y_idx in range(y_iters):#�ֱ���ɶ�������������������ȡ
            y_mid = int((y_iters - y_idx) * 5951 / y_iters)#������������ֵ��0 - 5951��û��496������ȡһ��
            '''����box���ĸ���������'''
            y_min = y_mid - search_range
            y_max = y_mid + search_range
            x_min = start_x - search_range
            x_max = start_x + search_range
            found_indices = ((lane_pixels_x >= x_min) & (lane_pixels_x <= x_max) & (lane_pixels_y >= y_min) & (
                        lane_pixels_y <= y_max)).nonzero()[0]#ȡ�������ߺ��������λ��
            found_x = lane_pixels_x[found_indices]#ȡ�������ߺ�������
            if len(found_x) > 1:
                start_x = int(np.mean(found_x))#ȥ�����ߺ��������ƽ��ֵ��Ϊʵ�ʺ�������
                flag = True#box�д��ڳ�����
            if flag:
                xyaxis[lane_idx].append([start_x, y_mid])#��������������ӵ�list��
            else:
                pass
    return xyaxis
def polyfit(xyaxis):
    left, right = xyaxis[0], xyaxis[1]
    left_x = [left[i][1] for i in range(len(left))]
    left_y = [left[i][0] for i in range(len(left))]
    right_x = [right[i][1] for i in range(len(right))]
    right_y = [right[i][0] for i in range(len(right))]
    left_fit_para, right_fit_para = np.polyfit(left_x, left_y, 2), np.polyfit(right_x, right_y, 2)
    return left_fit_para, right_fit_para
def draw(orign,predict_img, left_fit, right_fit):
    M,Minv = perspective()
    warped = warp(np.uint8(predict_img * 255))
    cv2.imwrite('./newwarp.jpg', warped)
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    plot_y = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
    right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]
    middle_fit_x = (left_fit_x+right_fit_x)/2
    dy = 2*left_fit[0]*710 + left_fit[1]
    ddy = 2*left_fit[0]
    R = ((1+dy**2)**(3/2))/ddy*2.87/584
    pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
    pts_middle = np.array([np.flipud(np.transpose(np.vstack([middle_fit_x, plot_y])))])
    PTS =np.hstack(pts_middle)
    pts = np.hstack((pts_left, pts_right))
    try:
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0),)
        cv2.polylines(color_warp,np.int_([PTS]),isClosed=False,color=(0,0,255),thickness=5)
        cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(255, 150,0),thickness=8)
    except:
        pass
    cv2.imwrite('./newwarp0.jpg', color_warp)
    newwarp = cv2.warpPerspective(color_warp, Minv, (880, 246))
    cv2.imwrite('./newwarp1.jpg',newwarp)
    newwarp = cv2.copyMakeBorder(newwarp,420,54,200,200, cv2.BORDER_CONSTANT,value=[0,0,0])
    result = cv2.addWeighted(orign, 1, newwarp, 0.3, 0)
    cv2.imwrite('./result.jpg',result)
    try:
        cv2.putText(result, str('Radius of curvature:') + str(round(R, 2)), (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 5), 1)
        print( str('Radius of curvature:') + str(round(R, 2))+'m')
    except:
        pass
    #offset center
    left_point = left_fit[0] * 700 ** 2 + left_fit[1] * 700 + left_fit[2]
    right_point = right_fit[0] * 700 ** 2 + right_fit[1] * 700 + right_fit[2]
    lane_center = (left_point + right_point) / 2
    off_center = (result.shape[1] / 2 - lane_center) * (3 / 734)
    off_center = round(off_center, 3)
    if off_center < 0:
        print("Offset to right:", abs(off_center), str("m"))
        center_messages = "Offset to right: : " + str(abs(off_center)) + str("m")
    if off_center > 0:
        print("Offset to left:", abs(off_center), str("m"))
        center_messages = "Offset to left:" + str(abs(off_center)) + str("m")
    if off_center == 0:
        print("On the center:", abs(off_center), str("m"))
    cv2.putText(result, center_messages, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    return result
def VideoProcee(input_path,output_path,timeF,fps):
    '''
    :param input_path: ��Ƶ����·��
    :param output_path: ��Ƶ���·��
    :param timeF: ���Ƶ��/֡
    :return: 0
    '''
    cap = cv2.VideoCapture(input_path)
    ret,frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    outvideo = cv2.VideoWriter(output_path,fourcc,fps,size)
    Nn =0
    while ret :
        ret,frame =  cap.read()
        try:
            if Nn % 5 == 0:
                frame = main(frame)
                cv2.imshow("final_lane", cv2.resize(frame, dsize=(640, 360)))
                key = cv2.waitKey(1)
                if key == 27:
                    cv2.destroyAllWindows()
                    break
            else:
                frame = frame
        except:
            frame=frame
        outvideo.write(frame)
        Nn =Nn+1
    cap.release()
    outvideo.release()
def main(img_arr):
    start = time.time()
    img = prediction(img_arr)
    img = output_process(img)
    xyaxis = find_lane_xyaxis(img)
    left_fit_para, right_fit_para = polyfit(xyaxis)
    final_img = draw(img_arr, img, left_fit_para, right_fit_para)
    end = time.time()
    print('Time:',round(end - start,2),'s')
    return final_img
if __name__ == "__main__" :
# Single image process
    filename = selectfile()
    img_arr  = cv2.imread(filename)
    result = main(img_arr)
    cv2.imshow('result',result)
    cv2.waitKey(0)
#  Video process
#     VideoProcee(input_path='./lane_video.mp4',output_path='./lane_detect.avi',timeF=1,fps=30)
    train()
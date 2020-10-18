import tensorflow as tf
import numpy as np
import cv2
def cnn_lane(sess,imgarr):
    imgarr = np.reshape(imgarr, newshape=[1, 60, 107, 3])
    predict = sess.run(y, feed_dict={input_x: imgarr, keep_prob: 1.0})[0]
    return predict
sess = tf.Session()
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


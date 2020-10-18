import numpy as np
import cv2
from glob import glob
import tensorflow as tf

light_name={1:"绿灯",2:"红灯",3:"黄灯"}
labels = np.array([[1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3]])
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]
y_one_hot = convert_to_one_hot(labels,4)
y_one_hot= np.delete(y_one_hot,[0],axis=1)
def wight_varible(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)#生成一个阶段的正太分布
    return tf.Variable(initial)
#初始化偏置
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')



#此处遍历训练文件不能使用os.listdir更不能使用glob，文件顺序和标签不对应了
path = './traffic_light'
img_arr = []
for i in range(30):
    fp = path+str('/light')+str(i)+str('.jpg')
    img = cv2.imread(fp)
    img = cv2.resize(img,(90,140))
    img_arr.append(img)
X = np.stack(img_arr) #(59, 140, 90, 3)
X = np.float32(X)
x = tf.placeholder(tf.float32,shape=[None,140,90,3])
y = tf.placeholder(tf.float32,shape=[None,3])

W_conv1=wight_varible([5,5,3,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

W_conv2=wight_varible([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

W_conv3=wight_varible([5,5,64,64])
b_conv3=bias_variable([64])
h_conv3=tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)
h_pool3=max_pool_2x2(h_conv3)

W_conv4=wight_varible([5,5,64,32])
b_conv4=bias_variable([32])
h_conv4=tf.nn.relu(conv2d(h_pool3,W_conv4)+b_conv4)
h_pool4=max_pool_2x2(h_conv4)
h_pool4_flat = tf.reshape(h_pool4,shape=[-1,9*6*32])#(?, 1728)

#全连接层
W_fc1=wight_varible([9*6*32,1024])
b_fc1=bias_variable([1024])
hfc1 = tf.nn.relu(tf.matmul(h_pool4_flat,W_fc1)+b_fc1)
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(hfc1,keep_prob)

W_fc2=wight_varible([1024,3])
b_fc2=bias_variable([3])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

train_step = tf.train.AdamOptimizer(1e-8).minimize(cross_entropy)

correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

print("x",X.shape)
print("y",y_one_hot.shape)
# saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        sess.run(train_step,feed_dict={x:X,keep_prob:0.7,y:y_one_hot})
        acc = sess.run(accuracy,feed_dict={x:X,keep_prob:1,y:y_one_hot})
        print("epoch:", epoch)
        print("acc:",acc)
    # saver.save(sess,"./traffic_light/model/CNN_net.ckpt")
    prediction = sess.run(prediction, feed_dict={x: X, keep_prob: 1})
    print(prediction)


# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     saver.restore(sess,'"./traffic_light/model/CNN_net.ckpt"')
#     sess.run(prediction,feed_dict={x:X,keep_prob:1,})
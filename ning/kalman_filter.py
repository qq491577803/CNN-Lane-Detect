import numpy
import pylab
import matplotlib.pyplot as plt
#kalman filter
#离散控制过系统：X(K) = AX(K-1)+BU(K)+W(K)----预测方程
#系统的测量值：Z(K) = HX(K) + V(K)------------测量值
#kalman filter的五个基本方程
#X(K/K-1) = A X(K-1/K-1) + BU(K) + W(K) ------------1
#P(K/K-1) = A P(K-1/K-1)A' + Q ---------------2
#X(K/K) = X(K/K-1) + Kg(K)(Z(K)-H X(K/K-1))---3
#Kg = P(K/K-1)H'/(H P(K/K-1)H' + R) ----------4
#P(K/K) = (I - Kg(K)H)P(K/K-1) ---------------5
#其中1,2现在状态的预测结果
#3系统的最优估计值，4是计算当前kalman增益，5是当前状态下X（K/K）的协方差。

#假设A=1 ，H =1
n_iter = 50
sz = (n_iter,)
x = -0.37727#x真实值
z = numpy.random.normal(x,0.1,size=sz)#系统的实际观测值（真实值加上噪声）
x_axis = [i+1 for i in range(len(z))]
xwhat = numpy.zeros(sz)#x后验估计值-----X(K/K)
P = numpy.zeros(sz)#后验估计误差-----P(K/K)
xwhatminus = numpy.zeros(sz)#先验估计值--X(K/K-1)
Pminus = numpy.zeros(sz)#先验估计误差---P(K/K-1)
K = numpy.zeros(sz)#kalman增益
R = 0.1**2#测量噪声的方差
Q = 1e-5#过程的协方差
#intial guesses
xwhat[0] = z[0]
P[0] = 1.0
for k in range(1,n_iter):
    #更新X(K/K-1),P(K/K-1)
    xwhatminus[k] = xwhat[k-1]   #X(K/K-1) = A X(K-1/K-1) + BU(K) + W(K),A = 1,BU(K) = 0
    Pminus[k] = P[k-1] + Q #P(K/K-1) = A P(K-1/K-1)A' + Q
    #测量值更新
    K[k] = Pminus[k]/(Pminus[k]+R) #Kg = P(K/K-1)H'/(H P(K/K-1)H' + R),H=1
    xwhat[k] = xwhatminus[k] + K[k]*(z[k] - xwhatminus[k])   #X(K/K) = X(K/K-1) + Kg(K)(Z(K)-H X(K/K-1))
    P[k] = (1-K[k])*Pminus[k]   #P(K/K) = (I - Kg(K)H)P(K/K-1)
plt.plot(x_axis,z)
# plt.plot(x_axis,xwhatminus)#
plt.plot(x_axis,xwhat)#滤波后的值
plt.show()
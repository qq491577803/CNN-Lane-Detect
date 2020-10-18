#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:36:40 2018

@author: fantastic
"""
# ! /usr/bin/python
# -*- coding: utf-8 -*-
import socket
import serial
import time
import thread
import threading
import random
import datetime
from Autocar_Function import stop_function_2,curve_function,right_angle_function,hire_function,triple_angle_function,lane_keeping,lane_wandao,changelane_function,stop_function_1
from multiprocessing import Process,Queue
#import queue
import math
import picamera
import numpy as np
import cv2
import numpy as np
import io
from picamera.array import PiRGBArray
from draw_line import auto_lane

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1);  # open named port at 9600,1s timeot
# sock = socket.socket(socket.AF_INET, socket.SOCK_STSREAM)
# server_address = ("localhost", 12422)
#server_address = ("172.20.10.6", 12345)
# print "connecting to %s:%s" % server_address, port
#sock.connect(server_address)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
'''
def client_to_sever(response):
    if response == '':
        print('mess is None')
    else:
        print "Sending '%s'" % response
        # onnetion,client_address = sock.accept()
        sock.send(response)

        recv = sock.recv(1024)
        print('secver_to_client_mess: ',recv)
    return recv
'''


def client_to_ardiuno(recv):
    print recv
    if recv == '':
        print('1223')
        return ''
    else:
        ser.write(recv);  # writ a string to port
        response = ser.readall( );  # read a string from port
        print('ardiuno_to_client_mess: ',response)
        return response

class MyThread(threading.Thread):
    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
def send_mess():
    mess = raw_input("response =")
    return mess

#duo jincheng
def receivedata_from_client(q):
    # print (data_recv)
    print "waiting .........."
    connection,client_address = sock.accept()
    print  "Connection from ", client_address
    while(1):
        data_recv = connection.recv(1024)
        print 'cor ----------------******time:',datetime.datetime.now()
        global data_recv
        print data_recv
        # print ('222')
        #a = q_2.get()
        connection.send(data_recv)
        #time.sleep(0.)
        q.put(data_recv)

#drive arduino
def drive_program(steer,speed,direc,bueezr):
    steer = steer - 30
    framehead=chr(100)
    steer_byte= chr(steer)
    speed_byte=chr(speed)
    direc_byte=chr(direc)
    buzzer_byte=chr(bueezr)
    frameend=chr(101)
    send_buf=framehead+steer_byte+speed_byte+direc_byte+buzzer_byte+frameend
    #if send_buf  >= 128 :
       # pass
##        send_buf = chr(126)
    print 'message',framehead,speed_byte,direc_byte,buzzer_byte,frameend,send_buf
    ser.write(send_buf.encode())
#raed data from port
def ardiuno_to_pi():
    response = ''   
    #response = ser.read(8)#read a string from port ser.read(10)
    #while (response == ''):
        #response = ser.read(8)
    # print response
    r_0 =0
    target_distance1=0
    target_distance2=0
    inf_01=0
    inf_02=0
    inf_03=0
    inf_04=0
    yaw=0
    r_7=0
    request=chr(127)
    ser.write(request.encode())
    
    response = ser.read(8)
    while response =='':
        ser.write(request.encode())#recvive data from ardiuno,lidar data
        ser.write(request.encode())
        '''
        a=0
        response_data = []
        while ser.inWaiting > 0:
            response_data.append(ser.read(8))
            print 'response_data888888888888888',response_data[a]
            a = a+1
        response = response_data[-1]
        '''
        response = ser.read(8)
        print 'jjjjjjjjjjjjjj'
    str(response)
    r_0 = response[0]
    r_7 = response[7]
    target_distance1 = ord(response[1])
# print target_distance1
    target_distance2 = ord(response[2])
# print target_distance2
    inf_sum1 = ord(response[3])
    inf_01=inf_sum1/10
    inf_02=inf_sum1%10
    inf_sum2 = ord(response[4])
    inf_03=inf_sum2/10
    inf_04=inf_sum2%10
    yaw_high = ord(response[5])
    yaw_low = ord(response[6])
    yaw = (yaw_high*256.0+yaw_low)/10.0-180.0
    return r_0,target_distance1,target_distance2,inf_01,inf_02,inf_03,inf_04,yaw,r_7


def main_function():
    date_recv = '0'
    last_data_recv = '0'
    stream = io.BytesIO()
    camera = picamera.PiCamera()
    w, h = 640, 360
    camera.resolution = (w, h)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=(w, h))
    size = (w, h)
    #first_recv = sock.recv(1024)
    # print('first_recv:',first_recv)
    #error = 0  # camera piancha
            ###excutive module
    curve_durationtime = 2.8
    right_angle_durationtime = 2.0
    hire_durationtime = 10.9
    triple_angle_durationtime = 9.2
    stop_durationtime = 1.5
    changelane_durationtime = 4.6
    hire_trigetimes = 0
    triple_trigetimes = 0
    park_trigetimes = 0
    steer_middle_position = 90
    zebracross_times = 0 
    '''
    steer = 90  # duo ji
    speed = 25  # motor speed
    direc = 1  # motor direction
    buzzer = 0
    a = 1
    '''
    last_error = 0
    last_steer = steer_middle_position
    last_flag = 1
    flag = 1
    right_angle_starttime = datetime.datetime.now()
    for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
        st = time.time()
        image = frame.array
        
        start = time.time()
        result,error,Radius,lanetype,SpeedLimit,RedLight,GreenLight,HireSign,ParkSign,zebracrossing=auto_lane(image)
        print "VisionTotleTime:",time.time()-start
        
        print 'Lane_type:',lanetype
        # if error=="None"and abs(error)>60:
        # sss=60
        #### decision module
        #r_0, target_distance1, target_distance2, inf_01, inf_02, inf_03, inf_04, yaw, r_7 = ardiuno_to_pi()
        #data_from_ardiuno = ardiuno_to_pi() #get data from ardiuno
        #print 'jghfdjkfhdkjfjd00000000',target_distance1
        #b = str(b[0])+str(b[1])+str(b[2])+str(b[3])+str(b[4])+str(b[5])+str(b[6])+str(b[7])+str(b[8])
        #q_2.put(b)
        # connection.send(data_from_ardiuno)  #send data to client
        target_detect = 0
        target_distance1 = 100
        #print 'dfhjghhhffhggfggdgd78889',target_distance1
        if target_distance1 < 30:
            target_detect =1



        if (last_flag == 1 or last_flag == 0 or last_flag ==2) and lanetype == 1:
            #and (HireSign + ParkSign + RedLight + ZebraCrossing) == 0
            flag = 1
            zebracross_times = 0 
        if (last_flag == 1 or last_flag == 0 or last_flag ==2) and lanetype == 2:
            flag = 2
            
            #if last_flag != 2:
                #flag = 2
                #curve_starttime = datetime.datetime.now()
        if (last_flag == 1 or last_flag == 0 or last_flag ==2) and lanetype == 3:
            if last_flag != 3:
                flag = 3
                
                right_angle_starttime = datetime.datetime.now()
            
        if  HireSign == 1:#HireSign hire_trigetimes ==0
            if last_flag != 4:
                flag = 4
                hire_starttime = datetime.datetime.now()
                hire_trigetimes = 1
        if ParkSign == 1:
            if last_flag != 5:
                flag = 5
                park_starttime = datetime.datetime.now()
                park_trigetimes = 1
        #if TriangleSign == 1:
            #if last_flag != 6:
                #flag = 6
                #triple_angle_starttime = datetime.datetime.now()
        if zebracrossing == 1 and zebracross_times == 0: 
            if last_flag != 7:
               flag = 7
               zebracross_times=1
               stop_starttime = datetime.datetime.now()
##        if RedLight == 1:#
##            if last_flag != 8:
##                flag = 8
##                changelane_starttime = datetime.datetime.now()
        if RedLight == 1 or target_detect == 1:
            flag = 0
        currenttime = datetime.datetime.now()
        print 'final_flag7878778878flagflag',flag

        #if flag == 2:

            # timediff = currenttime - curve_starttime
            # dt = timediff.total_seconds()
            # if dt <= curve_durationtime:
            #     flag = 2
            # elif dt > curve_durationtime:
            #     flag = 1
        if flag == 3:
            timediff = currenttime - right_angle_starttime
            dt = timediff.total_seconds()
            print "lalalalalalalalalalal",dt
            if dt <= right_angle_durationtime:
                flag = 3
            elif dt > right_angle_durationtime:
                flag = 1
        if flag == 4:
            timediff = currenttime - hire_starttime
            dt = timediff.total_seconds()
            if dt <= hire_durationtime:
                flag = 4
            elif dt > hire_durationtime:
                flag = 1
        # if flag == 5:
        #    timediff = currenttime - right_angle_starttime
        #    dt = timediff.total_seconds()
        #    if dt <= park_durationtime:
        #        flag = 3
        #    elif dt > park_durationtime:
        #        flag = 1
        if flag == 6:
            timediff = currenttime - triple_angle_starttime
            dt = timediff.total_seconds()
            if dt <= triple_angle_durationtime:
                flag = 6
            elif dt > triple_angle_durationtime:
                flag = 1
        if flag == 7:
            timediff = currenttime - stop_starttime
            dt = timediff.total_seconds()
            if dt <= stop_durationtime:
                flag = 7
            elif dt > stop_durationtime:
                flag = 1
##        if flag == 8:
##            timediff = currenttime - changelane_starttime
##            dt = timediff.total_seconds()
##            if dt <= changelane_durationtime:
##                flag = 8
##            elif dt > changelane_durationtime:
##                flag = 1           
        last_flag = flag
        
        
        cv2.putText(result, str('final_flage:') + str(flag), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 5), 1)
        cv2.imshow('flage',result)
        cv2.waitKey(1)
        

        print('offset: ', error, 'R :', Radius)
        rawCapture.truncate(0)
                    
##
##        if q.empty():
##            data_recv=last_data_recv
##       
##        else:
##            data_recv = q.get()
        #if Lane_type==1:
         #   data_recv=1
        #elif Lane_type==2:
         #   data_recv=2
            
            
        print 'cur -------time:',datetime.datetime.now()
        #print data_recv,'1111111111111111145511111'
##        if flag==3:
##            flag=1


        if flag == 0:
            steer, speed, direc, buzzer = stop_function_1(90)
        elif flag ==1:
            steer, speed, direc, buzzer,last_error = lane_keeping(error,last_error)
        elif flag == 2:
            steer, speed, direc, buzzer,last_steer = lane_wandao(error, last_steer)
            last_steer = steer
        elif flag == 3:
            steer, speed, direc, buzzer = right_angle_function(right_angle_starttime, right_angle_durationtime)
        elif flag == 4:
            steer, speed, direc, buzzer = hire_function(hire_starttime, hire_durationtime)
        # elif flag == 5:
        #    steer, speed, dir, buzzer = park_function(park_starttime , park_durationtime)
        elif flag == 6:
            steer, speed, direc, buzzer = triple_angle_function(hire_starttime, hire_durationtime)
        elif flag == 7:
            steer, speed, direc, buzzer = stop_function_2(last_steer,stop_starttime, stop_durationtime)
        elif flag == 8:
            steer, speed, direc, buzzer = changelane_function(changelane_starttime, changelane_durationtime)   
        last_data_recv = data_recv
        last_steer = steer
        data_ardiuno = drive_program(steer, speed, direc, buzzer)
        print 'MainFunctionTime:',time.time() -st
# if __name__ == "__main__":
##sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
##sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
##server_address = ("172.20.10.6",12351)
##print "Staring up on %s:%s" %  server_address
##sock.bind(server_address)
##sock.listen(5)
##socket.SOL_SOCKET



global data_recv
data_recv = 's'
q = Queue()
#q_2 = queue.LifoQueue()

process1 = Process(target=receivedata_from_client,args=(q,))
process2 = Process(target=main_function)
 
#process1.start()
process2.start()


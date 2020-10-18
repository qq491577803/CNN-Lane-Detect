import time
import datetime
import math
steer_middle_position = 90#90
right_angle_limit =35
left_angle_limit =-60
def stop_function_1(last_steer):
    steer=steer_middle_position
    speed=0
    dir=0
    buzzer=1
    return steer,speed,dir,buzzer
def stop_function_2(last_steer,model_starttime,durationtime):
    currenttime = datetime.datetime.now()
    timediff = currenttime - model_starttime
    dt = timediff.total_seconds()#the unit is second
    timepoint1 = 1.5
    if dt>=0 and dt<timepoint1:
        steer=last_steer
        speed=0
        dir=0
        buzzer=1
    else:
        steer = steer_middle_position
        speed = 0
        dir = 1
        buzzer = 0
    return steer,speed,dir,buzzer
def initial_function():
    steer = steer_middle_position
    speed=18
    dir=1
    buzzer=0
    return steer,speed,dir,buzzer

def curve_function(model_starttime,durationtime):
    currenttime = datetime.datetime.now()
    timediff = currenttime - model_starttime
    dt = timediff.total_seconds()#the unit is second
    timepoint1 = 2.8
    if dt>=0 and dt<timepoint1:
        steer = steer_middle_position + right_angle_limit - 5
        speed = 24
        dir = 1
        buzzer = 0
    else:
        steer = steer_middle_position
        speed = 0
        dir = 1
        buzzer = 0
    return steer, speed, dir, buzzer
def right_angle_function(model_starttime,durationtime):
    currenttime = datetime.datetime.now()
    timediff = currenttime - model_starttime
    dt = timediff.total_seconds()#the unit is second
    timepoint1 = 1.9 #turn left 0.4 second with -45 angle 
    timepoint2 = 2.0#turn right 3 second with 60 angle
    if dt>=0 and dt<timepoint1:
        steer = steer_middle_position + right_angle_limit + 3 #-45
        speed = 25
        dir = 1
        buzzer = 0
    elif dt>=timepoint1 and dt<timepoint2:
        steer = steer_middle_position #60
        speed = 18
        dir = 1
        buzzer = 0
    elif dt>timepoint2 and dt<durationtime:
        steer = steer_middle_position
        speed = 0
        dir = 1
        buzzer = 0
    else:
        steer = steer_middle_position
        speed = 0
        dir = 1
        buzzer = 0
    
    return steer, speed, dir, buzzer

def hire_function(model_starttime,durationtime):
    currenttime = datetime.datetime.now()
    timediff = currenttime - model_starttime
    dt = timediff.total_seconds()#the unit is second
    timepoint1 = 1.3 #turn right 1.3 second
    timepoint2 = 1.6 #go straight 0.3 second
    timepoint3 = 3.5 #turn left 1.9 second
    timepoint4 = 6.3 # stay stop for 2.8 second
    timepoint5 = 7.8 # continue turning left 1.5 second
    timepoint6 = 10.5 #turn right 2.7 second
    timepoint7 = 10.9 #return 0.4 second
    if dt>=0 and dt<timepoint1:
        steer = steer_middle_position + right_angle_limit -5#turn right
        speed = 25
        dir = 1
        buzzer = 0
    elif dt>=timepoint1 and dt<timepoint2:
        steer = steer_middle_position#drive forward
        speed = 18
        dir = 1
        buzzer = 0
    elif dt>=timepoint2 and dt<timepoint3:
        steer = steer_middle_position +left_angle_limit +8#turn left
        speed = 22#22
        dir = 1
        buzzer = 0
    elif dt>=timepoint3 and dt<timepoint4:
        steer = steer_middle_position#stop and wait passager
        speed = 0
        dir = 1
        buzzer = 0
    elif dt>=timepoint4 and dt<timepoint5:
        steer = steer_middle_position +left_angle_limit +8#turn left and leave station
        speed = 23#23
        dir = 1
        buzzer = 0
    elif dt>=timepoint5 and dt<timepoint6:
        steer = steer_middle_position + right_angle_limit -5#turn right and turn staight
        speed = 18
        dir = 1
        buzzer = 0
    elif dt>=timepoint6 and dt<timepoint7:
        steer = steer_middle_position#drive forward
        speed = 18
        dir = 1
        buzzer = 0
    else:
        steer = steer_middle_position#stop
        speed = 0
        dir = 1
        buzzer = 0
    return steer, speed, dir, buzzer
def changelane_function(model_starttime,durationtime):
    currenttime = datetime.datetime.now()
    timediff = currenttime - model_starttime
    dt = timediff.total_seconds()#the unit is second
    timepoint1 = 0.8 #turn right 0.8 second
    timepoint2 = 1.2 #go straight 0.4 second
    timepoint3 = 1.8 #turn left 0.6 second
    timepoint4 = 2.4# go straight 0.6 second
    timepoint5 = 3.2 # turn left 0.8 second
    timepoint6 = 3.8 #go straight 0.6 second
    timepoint7 = 4.6 #return 0.8 second
    if dt>=0 and dt<timepoint1:
        steer = steer_middle_position + right_angle_limit -5#turn right
        speed = 23
        dir = 1
        buzzer = 0
    elif dt>=timepoint1 and dt<timepoint2:
        steer = steer_middle_position#drive forward
        speed = 18
        dir = 1
        buzzer = 0
    elif dt>=timepoint2 and dt<timepoint3:
        steer = steer_middle_position +left_angle_limit +8#turn left
        speed = 22#22
        dir = 1
        buzzer = 0
    elif dt>=timepoint3 and dt<timepoint4:
        steer = steer_middle_position#stop and wait passager
        speed = 20
        dir = 1
        buzzer = 0
    elif dt>=timepoint4 and dt<timepoint5:
        steer = steer_middle_position +left_angle_limit +8#turn left 
        speed = 23#23
        dir = 1
        buzzer = 0
    elif dt>=timepoint5 and dt<timepoint6:
        steer = steer_middle_position # and turn staight
        speed = 18
        dir = 1
        buzzer = 0
    elif dt>=timepoint6 and dt<timepoint7:
        steer = steer_middle_position + right_angle_limit -5#turn right
        speed = 23
        dir = 1
        buzzer = 0
    else:
        steer = steer_middle_position#stop
        speed = 0
        dir = 1
        buzzer = 0
    return steer, speed, dir, buzzer
def triple_angle_function(model_starttime,durationtime):
    currenttime = datetime.datetime.now()
    timediff = currenttime - model_starttime
    dt = timediff.total_seconds()#the unit is second
    timepoint1 = 2.8#turn right 2.8 second
    timepoint2 = 3.6 #go straight 0.8 second
    timepoint3 = 6 #turn left 2.4 second
    timepoint4 = 7.4# go straight 1 second
    timepoint5 = 8.6#turn right 1.2 second
    if dt>=0 and dt<timepoint1:
        steer = steer_middle_position + right_angle_limit -5#turn right
        speed = 26
        dir = 1
        buzzer = 0
    elif dt>=timepoint1 and dt<timepoint2:
        steer = steer_middle_position  #stay straight
        speed = 18
        dir = 1
        buzzer = 0
    elif dt>=timepoint2 and dt<timepoint3:
        steer = steer_middle_position + left_angle_limit +8#turn left
        speed = 26
        dir = 1
        buzzer = 0
    elif dt>=timepoint3 and dt<timepoint4:
        steer = steer_middle_position #stay straight
        speed = 18
        dir = 1
        buzzer = 0
    elif dt>=timepoint4 and dt<timepoint5:
        steer = steer_middle_position + right_angle_limit -5 #turn right
        speed = 26
        dir = 1
        buzzer = 0
    else:
        steer = steer_middle_position#stop
        speed = 0
        dir = 1
        buzzer = 0
    return steer, speed, dir, buzzer
def park_function(model_starttime,durationtime):
    currenttime = datetime.datetime.now()
    timediff = currenttime - model_starttime
    dt = timediff.total_seconds()#the unit is second
    timepoint1 = 3.0 #turn right 3.0 second with -35 angle 
    timepoint2 = 4.0 #park vertical
    if dt>=0 and dt<timepoint1:
        steer = steer_middle_position + right_angle_limit #-60
        speed = 25
        dir = 2
        buzzer = 0
    elif dt>=timepoint1 and dt<timepoint2:
        steer = steer_middle_position #
        speed = 18
        dir = 2
        buzzer = 0
    elif dt>timepoint2 and dt<durationtime:
        steer = steer_middle_position
        speed = 0
        dir = 1
        buzzer = 0
    else:
        steer = steer_middle_position
        speed = 0
        dir = 1
        buzzer = 0
    
    return steer, speed, dir, buzzer
def lane_keeping(error,last_error):

    threshold = 10  # yuzhi
    depth = 270  # calibration depth280
    steer=90
    speed=18 #18
    direc=1
    buzzer=0

    if error == "None":  # left lane lose
        steer = steer_middle_position - 8#10
        #data_ardiuno = drive_program(theta, speed, direc, buzzer)  #
        print "1", steer, speed
    elif abs(error) >= threshold:  # xunji alogthrim
        if error > 0:  # rigth steer
            steer = steer_middle_position - (int)(1.4 * (math.atan2(error, depth) / math.pi * 180) - 2)  # 1.4 -2duo ji zhuanjiao
        else:  # left steer
            steer = steer_middle_position - (int)(1.3* (math.atan2(error, depth) / math.pi * 180) + 3)  # 1.3duo ji zhuanjiao

        
        # data_ardiuno = drive_program(theta, speed, direc, buzzer)  # buzzer
        print "2", steer, error
        '''
    else:  # normal alogthrim
        if error - last_error > 0:
            steer = steer_middle_position-10
            # data_ardiuno = drive_program(steer, speed, direc, buzzer)
        else:
            steer = steer_middle_position+5
            # data_ardiuno = drive_program(steer, speed, direc, buzzer)
        print "3", steer, speed
        '''
    if error != 'None':
        last_error = error
    else:
        last_error = 0
    print "zhidaozhidaozhidaozhidaozhidaozhidaozhidaozhidaozhidaozhidaozhidaozhidaozhidao"
    if steer >= steer_middle_position+right_angle_limit:
        steer = steer_middle_position+right_angle_limit
    if steer < steer_middle_position+left_angle_limit:
        steer = steer_middle_position+left_angle_limit
    return steer, speed, direc, buzzer,last_error
def lane_wandao(error,last_steer):
    
    threshold = 10  # yuzhi
    depth = 280  # calibration depth
    steer=90
    speed=20
    direc=1
    buzzer=0
    if error=="None":#left lane lose
      #steer=last_steer
      steer=steer_middle_position+40
      print "1",steer,speed
                
    elif abs(error)>=threshold:  #xunji alogthrim
         if error>0:   #rigth steer              
             #error=-error
             steer = steer_middle_position - (int)(2.3 * (math.atan2(error, depth) / math.pi * 180) +1)
             #steer=last_steer
             
         else:            #left steer
             steer = steer_middle_position-(int)(2.3*(math.atan2(error,depth)/math.pi*180)+5) #2.5duo ji zhuanjiao
    if steer >= steer_middle_position+right_angle_limit:
        steer = steer_middle_position+right_angle_limit
    if steer < steer_middle_position+left_angle_limit:
        steer = steer_middle_position+left_angle_limit
         #data_ardiuno = drive_program(theta, speed, direc, buzzer)#buzzer
    print "2",steer,error
    last_steer=steer
    print "wandaowandaowandaowandaowandaowandaowandaowandaowandaowandaowandao"
    return steer, speed, direc, buzzer,last_steer
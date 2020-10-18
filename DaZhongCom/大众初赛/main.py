import cv2
import numpy as np
import colorsys
import random
from keras.preprocessing import image
from sklearn import svm
from sklearn.externals import joblib
from moviepy.video.io.VideoFileClip import VideoFileClip
from sklearn.metrics import accuracy_score
from camera_cal import undistort
import MRCNN
from draw_lane import draw_lane
from direction_distance import direction_calculate
from direction_distance import distance_calculate
from direction_distance import off_center
from TrafficLightThreshold import light_recongize

class InferenceConfig():
    #批次大小设置为1，一次一张图像。 批量大小= GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BACKBONE_SHAPES=np.array([[256, 256], [128, 128],  [ 64,  64],  [ 32,  32],  [ 16,  16]])
    BACKBONE_STRIDES=[4, 8, 16, 32, 64]
    BATCH_SIZE=1
    BBOX_STD_DEV=[ 0.1,  0.1,  0.2,  0.2]
    DETECTION_MAX_INSTANCES=100
    DETECTION_MIN_CONFIDENCE=0.6 #0.5
    DETECTION_NMS_THRESHOLD=0.3
    IMAGE_MAX_DIM=1024
    IMAGE_MIN_DIM=800
    IMAGE_PADDING=True
    IMAGE_SHAPE=np.array([1024, 1024,    3])
    LEARNING_MOMENTUM=0.9
    LEARNING_RATE =0.002
    MASK_POOL_SIZE=14
    MASK_SHAPE    =[28, 28]
    MAX_GT_INSTANCES=100
    MEAN_PIXEL      =[ 123.7,  116.8,  103.9]
    MINI_MASK_SHAPE =(56, 56)
    NAME            ="coco"
    NUM_CLASSES     =81
    POOL_SIZE       =7
    POST_NMS_ROIS_INFERENCE =1000
    POST_NMS_ROIS_TRAINING  =2000
    ROI_POSITIVE_RATIO=0.33
    RPN_ANCHOR_RATIOS =[0.5, 1, 2]
    RPN_ANCHOR_SCALES =(32, 64, 128, 256, 512)
    RPN_ANCHOR_STRIDE =2
    RPN_BBOX_STD_DEV  =np.array([ 0.1,  0.1,  0.2 , 0.2])
    RPN_TRAIN_ANCHORS_PER_IMAGE=256
    RPN_NMS_THRESHOLD = 0.3
    STEPS_PER_EPOCH            =1000
    TRAIN_ROIS_PER_IMAGE       =128
    USE_MINI_MASK              =True
    USE_RPN_ROIS               =True
    VALIDATION_STPES           =50
    WEIGHT_DECAY               =0.0001
class_names = np.load('./configs/coco_class_names.npy')
def car_detection(neural_net, input_img):
    img = cv2.resize(input_img, (1024, 1024))
    img = image.img_to_array(img)
    results = neural_net.detect([img], verbose=0)
    r = results[0]
    boxes = r
    for class_num in range(len(r["class_ids"])):
        class_id = r["class_ids"][class_num]

    final_img = draw_message(input_img, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
    inp_shape = image.img_to_array(input_img).shape
    final_img = cv2.resize(final_img, (inp_shape[1], inp_shape[0]))
    return final_img,boxes
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image
def draw_message(input_image, boxes, masks, class_ids, class_names,scores=None):
    N = boxes.shape[0]
    if not N:
        print("No object has been detected !")

        return input_image
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    img0 = cv2.resize(input_image, (1024, 1024))
    img = image.img_to_array(img0)

    colors = random_colors(N)
    masked_image = img.astype(np.uint32).copy()
    for i in range(N):
        if class_ids[i] == 3 or class_ids[i] == 8 or class_ids[i] ==2 or class_ids[i] ==1  or class_ids[i] ==6  or class_ids[i] == 10:
        # if class_ids[i] == 3 or class_ids[i] == 8 or class_ids[i] ==6  :

            color = colors[i]
            if not np.any(boxes[i]):
                continue
            y1, x1, y2, x2 = boxes[i]

            # 信号灯识别
            if class_ids[i] == 10:
                img_arr = img0[y1:y2,x1:x2]
                img_arr = np.uint8(img_arr)
                img_arr = cv2.resize(img_arr,(90,140))
                '''SVM 信号灯识别
                light_name = {1: "Red", 2: "None", 3: "Green",4:'Yellow'}
                img = cv2.resize(img_arr, (90, 140))
                img = np.reshape(img, [1, 140 * 90*3])
                clf = joblib.load("./model/light.model")
                pre_num = int(clf.predict(img))
                name = light_name[pre_num]
                print(name)
                '''
                #阈值法信号灯识别
                message = light_recongize(img_arr)
                print("信号灯：",message)
                try:
                    masked_image = cv2.putText(masked_image.astype(np.uint8), message,(x1 - 5, y1 - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                except:
                    masked_image = cv2.putText(masked_image.astype(np.uint8), message,(x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                masked_image = cv2.putText(masked_image.astype(np.uint8), message,(425,980), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            else:
                pass
            masked_image = masked_image.astype(np.uint8)
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), (150,255,150), 2)
            masked_image = masked_image.astype(np.uint32)
            # Label
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = class_names[class_id]
            try:
                masked_image = cv2.putText(masked_image.astype(np.uint8),str(label)+" acc:"+str(round(score,2)),(x1-5,y1-5),font,0.5,(0,0,255),1)
            except:
                masked_image = cv2.putText(masked_image.astype(np.uint8),str(label)+" acc:"+str(round(score,2)),(x1,y1),font,0.5,(0,0,255),1)

            #绘制目标车辆方位和距离
            y,x =np.int(y2),np.int((x1+x2)/2)
            flag,alpha,n,m,raw,colum = direction_calculate(x,y)
            if flag == 1:
                distance = distance_calculate(n, m, raw, colum)
                print("目标车量的方位：{} 距离：{}".format(alpha,distance))
                try:
                    masked_image = cv2.putText(masked_image.astype(np.uint8), "A:"+str(alpha)+"deg" + " L:" + str(round(distance, 2))+"m",(x1,y2 + 15), font, 0.5, (0, 0, 255), 1)
                except:
                    masked_image = cv2.putText(masked_image.astype(np.uint8),"A:" + str(alpha) + "deg" + " L:" + str(round(distance, 2)) + "m", (x1, y2), font, 0.5, (0, 0, 255), 1)
            else:
                pass
            # Mask
            if class_ids[i] == 10:
                pass
            else:
                mask = masks[:, :, i]
                masked_image = apply_mask(masked_image, mask, color)
                padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
                padded_mask[1:-1, 1:-1] = mask
    return masked_image.astype(np.uint8)

#加载全局变量
NN = MRCNN.MaskRCNN( model_dir='logs')
NN.load_weights('./model/mask_rcnn.model', by_name=True)
config = InferenceConfig()

#视频处理
def processVideo(input_path,output_path,detection_method,audio=False):
    video = VideoFileClip(input_path)
    output_video = video.fl_image(detection_method)
    output_video.write_videofile(output_path, audio=False)
#红绿灯视频处理
def LightVideoProcee(input_path,output_path,timeF,fps):
    '''    
    :param input_path: 视频输入路径
    :param output_path: 视频输出路径
    :param timeF: 检测频率/帧
    :return: 0
    '''
    cap = cv2.VideoCapture(input_path)
    ret,frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    outvideo = cv2.VideoWriter(output_path,fourcc,fps,size)
    Nn =0
    while ret :
        print(ret)
        ret,frame =  cap.read()
        try:
            if Nn % timeF ==0:
                frame = lightRecoginze(frame)
                # cv2.imshow('img', frame)
                # cv2.waitKey(1)
            else:
                frame = frame
        except:
            frame=frame
        outvideo.write(frame)
        Nn =Nn+1
    cap.release()
    outvideo.release()
#红绿灯识别
def lightRecoginze(img_arr):
    img_arr = cv2.resize(img_arr, (1280, 720))
    # img_arr = undistort(img_arr)
    print("processing")
    object_detect ,boxes=car_detection(NN,img_arr)
    result =object_detect
    print("Rconginze Successde !")
    cv2.imshow("light",result)
    cv2.waitKey(1)
    return result
#车道检测
def lane_detection(img_arr):
    img_arr = cv2.resize(img_arr, (1280, 720))
    print("Processing ...waitting...")
    drawlane,left_fit,right_fit = draw_lane(img_arr)
    offset = off_center(drawlane,left_fit,right_fit)
    result = cv2.putText(drawlane.astype(np.uint8),offset, (60,60),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 1)
    print("Detect Successed !")
    cv2.imshow("lane",result)
    cv2.waitKey(1)
    return result
#综合检测识别
def combine(img_arr):
    img_arr = cv2.resize(img_arr, (1280, 720))
    # img_arr = undistort(img_arr)
    try:
        print("Processing ...waitting...")
        drawlane,left_fit,right_fit = draw_lane(img_arr)
        offset = off_center(drawlane,left_fit,right_fit)
        result = cv2.putText(drawlane.astype(np.uint8),offset, (60,60),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 1)
        object_detect ,boxes=car_detection(NN,result)
        print("Rconginze Successde !")
        result = object_detect
    except:
        print('Reconginze Failed !')
        result = object_detect
    cv2.imshow('combine',result)
    cv2.waitKey(1)
    return result

if __name__ == "__main__":

    #车道线单张图片检测          处理不同的车道线，需要更改lane_detection中的perspective
    # result = lane_detection(cv2.imread('./input/test_lane.jpg'))
    # cv2.imshow('lane',result)
    # cv2.waitKey(0)
    # #车道线视频处理    注：运行到最后一帧会报错，但是没影响，视频照样生成
    # processVideo('./videos/input/video.mp4', './videos/output/video_lane.mp4', lane_detection, audio=False)
    ##车道线在线检测
    # cap =  cv2.VideoCapture('./videos/input/video.mp4')
    # ret,frame = cap.read()
    # while ret :
    #     ret,frame = cap.read()
    #     img = lane_detection(frame)
    #     cv2.imshow("lane",img)
    #     key = cv2.waitKey(5)
    #     if key == 27:  # Esc 停止
    #         break
    # cap.release()
    #车道线检测+车辆识别+行人识别+红绿灯识别+车道曲率+偏心 图片处理
    # com_arr = combine(cv2.imread('./input/test_combine.jpg'))
    # cv2.imshow("combine",com_arr)
    # cv2.waitKey(0)
    #目标识别车道线检测视频处理
    # processVideo('./videos/input/video.mp4', './videos/output/combine.mp4', combine, audio=False)

    #信号灯单张图片识别
    # light_arr = lightRecoginze(cv2.imread('./input/test_light.jpg'))
    # cv2.imshow("light",light_arr)
    # cv2.waitKey(0)
    #信号灯视频处理
    LightVideoProcee('./videos/input/light_original.mp4', './videos/output/lightvideo.avi', 1, 30)




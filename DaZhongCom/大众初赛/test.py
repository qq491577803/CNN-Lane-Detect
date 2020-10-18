import cv2

import cv2
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
outVideo = cv2.VideoWriter('saveDir.avi',fourcc,30,(1280,720))
n=1
pt = './11/'
while 1:
    path =  pt +str(n)+'.jpg'
    print(path)
    frame = cv2.imread(path)
    outVideo.write(frame)
    # cv2.imshow(',',frame)
    # cv2.waitKey(1)
    if n ==2098:
        break
    n =n+1
outVideo.release()


# import cv2
# fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
# outVideo = cv2.VideoWriter('saveDir.avi',fourcc,30,())
# while ret:
#     ret,fram = cap.read()
#     outVideo.write(fram)
#     cv2.imshow("ges",fram)
#     key = cv2.waitKey(5)
#     if key == 27:
#         break
# cap.release()
# outVideo.release()
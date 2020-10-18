import numpy as np
import cv2

def main(frame):
	red_val = 0
	green_val = 65
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	red_lower = np.array([red_val-10,100,100])
	red_upper = np.array([red_val+10,255,255])
	green_lower = np.array([green_val-10,100,100])
	green_upper = np.array([green_val+10,255,255])
	red_mask = cv2.inRange(hsv, red_lower, red_upper)
	green_mask = cv2.inRange(hsv, green_lower, green_upper)
	red_res = cv2.bitwise_and(frame,frame, mask= red_mask)
	green_res = cv2.bitwise_and(frame,frame, mask= green_mask)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	# Morphological Closing
	red_closing = cv2.morphologyEx(red_res,cv2.MORPH_CLOSE,kernel)
	green_closing = cv2.morphologyEx(green_res,cv2.MORPH_CLOSE,kernel)

	cv2.imshow("greenmask",green_closing)
	cv2.waitKey(0)
	#Convert to Black and White image
	red_gray = cv2.cvtColor(red_closing, cv2.COLOR_BGR2GRAY)
	green_gray = cv2.cvtColor(green_closing, cv2.COLOR_BGR2GRAY)
	(thresh1, red_bw) = cv2.threshold(red_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	(thresh2, green_bw) = cv2.threshold(green_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	cv2.imshow("greenbw",green_bw)
	cv2.waitKey(0)
	# Count pixel changes
	red_black = cv2.countNonZero(red_bw)
	print(red_black)
	if red_black > 20000:
		print ("RED")

	green_black = cv2.countNonZero(green_bw)
	print(green_black)
	if green_black > 18000:
		print ("GREEN")

if __name__ == "__main__":
	img  = cv2.imread("./lv.jpg")
	img = cv2.resize(img,dsize=(640,360))
	main(img)
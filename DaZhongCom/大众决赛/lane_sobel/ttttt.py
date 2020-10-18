import cv2

img = cv2.imread('./cal_image.jpg')
cv2.imshow("img",img)
img = cv2.resize(img,dsize=(320,180))
cv2.imwrite("./cal_resi_img.jpg",img)
cv2.imshow("img",img)
cv2.waitKey(0)
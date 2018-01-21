import cv2
import time
import numpy as np
cam = cv2.VideoCapture(1)
cam.set(3,1280)
cam.set(4,720)
time.sleep(2)
cam.set(15,-8.0)
while True:
	ret, img = cam.read()
	img = cv2.medianBlur(img,5)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, np.array([0,0,200]), np.array([180,20,255])) #+ cv2.inRange(hsv, np.array([170,100,100]), np.array([180,255,255]))
	output_img = img.copy()
	output_img[np.where(mask==0)] = 0
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None
	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
		# only proceed if the radius meets a minimum size
		if radius > 50:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(img, (int(x), int(y)), int(radius),(0, 255, 255), 2)
			cv2.circle(img, center, 5, (0, 0, 255), -1)
	cv2.imshow("img", img)
	k=cv2.waitKey(33)
	if k == 27:
		cv2.imwrite("test9.jpg", img)
		break
	else:
		continue



import cv2
import numpy as np
from PIL import Image
import pytesseract
import argparse
import os
import time

kernel = np.ones((5,5),np.uint8)
#cam = cv2.VideoCapture(1)
#cam.set(3,1280)
#cam.set(4,720)
#time.sleep(2)
#cam.set(15,-8.0)
#__, im = cam.read()
im=cv2.imread('board6.jpg')
cv2.imshow("img", im)
cv2.waitKey(0)
#im = cv2.resize(im, (1920, 1080))
#imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
imgray = (255-im)
blur = cv2.GaussianBlur(imgray,(3,3),0)
cv2.imshow("img", blur)
cv2.waitKey(0)
threshnew = cv2.inRange(blur, np.array([155,155,155]), np.array([255,255,255]))
cv2.imshow("img", threshnew)
cv2.waitKey(0)
#threshnew = (255-mask)
#imnext = cv2.dilate(thresh,kernel,iterations = 1)
#threshnew = cv2.erode(threshnew,kernel,iterations = 1)
cv2.imshow("img", threshnew)
cv2.waitKey(0)
__, contours, __ = cv2.findContours(threshnew,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print len(contours)	
count = 0
check=0
for cnt in contours:
	if check%2==0 or check%2==1:
		if cv2.contourArea(cnt) > 10 and cv2.contourArea(cnt) < 5000:
			print cv2.contourArea(cnt)
			count+=1
			print count
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
	check+=1

cv2.imshow("img", im)
cv2.waitKey(0)

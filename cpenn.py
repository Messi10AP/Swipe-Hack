import cv2
import numpy as np
from PIL import Image
import pytesseract
import argparse
import os
import time
from serial import Serial

strings = 'qwertyuiopasdfghjklzxcvbnm'

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)
	
def cropInitialImage(contours, coords):
	copy = []
	#print len(contours)
	#print len(coords)
	for i in range(len(coords)):
		copy.append(coords[i])
	copy.sort()
	#print copy
	return [coords.index(copy[0]), coords.index(copy[1]), coords.index(copy[2]), coords.index(copy[3])]

def getCurrentPosition():
	#print "test"
	img = cv2.medianBlur(img,5)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, np.array([0,255,255]), np.array([180,20,255])) #+ cv2.inRange(hsv, np.array([160,100,100]), np.array([180,255,255]))
	output_img = img.copy()
	output_img[np.where(mask==0)] = 0
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None
	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		return (x,y)
	else:
		return (0,0)
	

def letter(currposition):
	x = currposition[0]
	y = currposition[1]
	mindist = 999999999
	let = ''
	for i in coordinates:
		tx, ty = i
		tempdist = ((float(x)-float(tx))**2+(float(y)-float(ty))**2)**0.5
		#print tempdist
		if tempdist < mindist:
	  		mindist = tempdist
	    		let = letters[coordinates.index(i)]
	return let


coordinates = []
letters = []
#ser = Serial('/dev/ttyACM1', 9600)
cam = cv2.VideoCapture(1)
#cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cam.set(3,1280)
cam.set(4,720)
#cam.set(
#time.sleep(2)
#cam.set(15,8.0)
#try:11,11
#	s = ser.readline()
#except: 
#	continue
#time.sleep(5)

kernel = np.ones((11,11),np.uint8)

while True:	
	ret, im = cam.read()
	im = cv2.flip( im, 0 )
	im = cv2.flip( im, 1 )
	cv2.rectangle(im, (40,220), (1240,580), (255,0,0), 3)
	cv2.imshow("img", im)
	k=cv2.waitKey(33)
	if k == 27:
		break

#lab= cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
#l, a, b = cv2.split(lab)
#clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#cl = clahe.apply(l)
#limg = cv2.merge((cl,a,b))
#im = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


#a = np.double(im)
#b = a + 15
#im = np.uint8(b)
#im=cv2.imread('board6.jpg')
#im = cv2.resize(im, (1920, 1080))
im = adjust_gamma(im, 0.5)
im = im[220:580, 40:1240]
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
temp =  cv2.erode(imgray,kernel,iterations = 2)
#temp = cv2.GaussianBlur(temp,(5,5),0)
#temp = cv2.cvtColor(temp,cv2.COLOR_GRAY2RGB)
imgray = (255-imgray)

blur = cv2.GaussianBlur(imgray,(5,5),0)
#blur = (255-blur)
cv2.imshow("img", blur)
#cv2.waitKey(0)
#ret,thresh = cv2.threshold(blur,150,255,cv2.THRESH_BINARY)
ret3,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
thresh = cv2.dilate(thresh,kernel,iterations = 1)
thresh =  cv2.erode(thresh,kernel,iterations = 1)
#temp=thresh
cv2.imshow("img", thresh)
cv2.waitKey(0)
threshnew = (255-thresh)
#imnext = cv2.dilate(thresh,kernel,iterations = 1)
#threshnew = cv2.erode(threshnew,kernel,iterations = 1)
cv2.imshow("img", threshnew)
#cv2.waitKey(0)
__, contours, __ = cv2.findContours(threshnew,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

print len(contours)	
count = 0
check=0
actualcontours=[]
y_coords = []
x_coords = []
for cnt in contours:
	if cv2.contourArea(cnt) > 10 and cv2.contourArea(cnt) < 10000:
		actualcontours.append(cnt)
		
		#print cv2.contourArea(cnt)
		count+=1
		#print count
		x,y,w,h = cv2.boundingRect(cnt)
		y_coords.append(y)
		x_coords.append(x)
		cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
		#if y>50 and y+h < 660 and x>50 and x+w < 1230:
			#roi = im[y-50:y+h+50, x-50:x+w+50]
			#roi = (255-roi)
			#roi= cv2.resize(roi, (426, 240))
			#cv2.imshow("roi",roi)
			#cv2.waitKey(0)
			#cv2.imwrite("thresh.png",roi)
			#text = pytesseract.image_to_string(Image.open('thresh.png'))
			#print(text)
	check+=1
cv2.imshow('im', im)
cv2.waitKey(0)
#ret = cropInitialImage(temp, actualcontours, y_coords)
#print ret[1]
#print ret[0]
#x,y,w,h = cv2.boundingRect(actualcontours[ret[1]])
#cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,255),2)
#xx,yy,ww,hh = cv2.boundingRect(actualcontours[ret[3]])
#cv2.rectangle(im,(xx,yy),(xx+ww,yy+hh),(0,255,255),2)
q,a,z,w = cropInitialImage(contours, x_coords)	
#print q, a, z, w
#print "AS"
qx,qy,qw,qh = cv2.boundingRect(contours[q])
#print qx,qy,qw,qh, "Q"
#cv2.circle(im, (qx,qy), 45, (255,255,0))
cv2.imshow("img", im)
#cv2.waitKey(0)
ax,ay,aw,ah = cv2.boundingRect(contours[a])
zx,zy,zw,zh = cv2.boundingRect(contours[z])
wx,wy,ww,wh = cv2.boundingRect(contours[w])
DISTANCE = 128#wx-qx+16
#print DISTANCE
for i in range(10):
	center = (qx+qw/2+DISTANCE*i, qy+qh/2)
	coordinates.append(center)
	letters.append(strings[i])
for i in range(9):
	center = (ax+aw/2+DISTANCE*i, ay+ah/2)
	coordinates.append(center)
	letters.append(strings[10+i])
for i in range(7):
	center = (zx+zw/2+DISTANCE*i, zy+zh/2)
	coordinates.append(center)
	letters.append(strings[19+i])
#print coordinates
for i in coordinates:
	cv2.circle(im, i, 64, (255,255,0))
	#print letters[coordinates.index(i)]
	cv2.imshow("img", im)
	cv2.waitKey(1)
	#print letters
cam.release()
#cv2.imshow("img", im)
#cv2.waitKey(0)
cam2= cv2.VideoCapture(1)
cam2.set(3,1280)
cam2.set(4,720)
#time.sleep(2)
done = 0
word=''
start = False
while True:
	ret, img = cam2.read()
	img = cv2.flip( img, 0 )
	img = cv2.flip( img, 1 )
	img = img[220:580, 40:1240]
	imgg = cv2.medianBlur(img,5)
	hsv = cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV)
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
		#print radius
		if radius > 50:
			done=0
			start = True
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(img, (int(x), int(y)), int(radius),(0, 255, 255), 2)
			cv2.circle(img, center, 5, (0, 0, 255), -1)
			#print (x,y)
			r = letter((x,y))
 			cv2.putText(img,str(r),(int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,0,255),2,cv2.LINE_AA)
		#cv2.waitKey(0)
			word+=r
		else:
			done+=1
	cv2.imshow("img", img)
	if done>=20 and start:
		print word #process word
		word = ''
		done = 0
		start=False
		
	k=cv2.waitKey(	33)
	if k == 32:
		break
	else:
		continue



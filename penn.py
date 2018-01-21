import cv2
import numpy as np
from PIL import Image
import pytesseract
import argparse
import os
import time
import re
import collections
from itertools import product, imap
from serial import Serial
from Tkinter import *
from PIL import Image, ImageTk
strings = 'qwertyuiopasdfghjklzxcvbnm'
from matplotlib import pyplot as plt


VERBOSE = True
vowels = set('aeiouy')
alphabet = set('abcdefghijklmnopqrstuvwxyz')

##STRIP

def recognizeString(freq, str): 
	str = str + '@'
	match = str[0]
	count = 1
	final_str = ''
	for x in range (1, len(str)):
		if str[x] == match:
			count = count + 1
		else:
			#for y in range (0, count/freq):
			if count >= freq:
				final_str = final_str + str[x-1]
			count = 1
			match = str[x]
	return final_str

### IO

def words(text):
        """filter body of text for words"""
	return re.findall('[a-z]+', text.lower())

def train(text, model=None):
	"""generate or update a word model (dictionary of word:frequency)"""
	model = collections.defaultdict(lambda: 0) if model is None else model
	for word in words(text):
		model[word] += 1
	return model

def train_from_files(file_list, model=None):
	for f in file_list:
		model = train(file(f).read(), model)
	return model


### UTILITY FUNCTIONS

def numberofdupes(string, idx):
	"""return the number of times in a row the letter at index idx is duplicated"""
	# "abccdefgh", 2  returns 1
	initial_idx = idx
	last = string[idx]
	while idx+1 < len(string) and string[idx+1] == last:
		idx += 1
	return idx-initial_idx

def hamming_distance(word1, word2):
	if word1 == word2:
		return 0
	dist = sum(imap(str.__ne__, word1[:len(word2)], word2[:len(word1)]))
	dist = max([word2, word1]) if not dist else dist+abs(len(word2)-len(word1))
	return dist

def frequency(word, word_model):
    return word_model.get(word, 0)

### POSSIBILITIES ANALYSIS

def variants(word):
    """get all possible variants for a word"""
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts    = [a + c + b for a, b in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def double_variants(word):
    """get variants for the variants for a word"""
    return set(s for w in variants(word) for s in variants(w))

def reductions(word):
    """return flat option list of all possible variations of the word by removing duplicate letters"""
    word = list(word)
    # ['h','i', 'i', 'i'] becomes ['h', ['i', 'ii', 'iii']]
    for idx, l in enumerate(word):
        n = numberofdupes(word, idx)
        # if letter appears more than once in a row
        if n:
            # generate a flat list of options ('hhh' becomes ['h','hh','hhh'])
            flat_dupes = [l*(r+1) for r in xrange(n+1)][:3] # only take up to 3, there are no 4 letter repetitions in english
            # remove duplicate letters in original word
            for _ in range(n):
                word.pop(idx+1)
            # replace original letter with flat list
            word[idx] = flat_dupes

    # ['h',['i','ii','iii']] becomes 'hi','hii','hiii'
    for p in product(*word):
        yield ''.join(p)

def vowelswaps(word):
    """return flat option list of all possible variations of the word by swapping vowels"""
    word = list(word)
    # ['h','i'] becomes ['h', ['a', 'e', 'i', 'o', 'u', 'y']]
    for idx, l in enumerate(word):
        if type(l) == list:
            pass                        # dont mess with the reductions
        elif l in vowels:
            word[idx] = list(vowels)    # if l is a vowel, replace with all possible vowels

    # ['h',['i','ii','iii']] becomes 'hi','hii','hiii'
    for p in product(*word):
        yield ''.join(p)

def both(word):
    """permute all combinations of reductions and vowelswaps"""
    for reduction in reductions(word):
        for variant in vowelswaps(reduction):
            yield variant

### POSSIBILITY CHOOSING

def suggestions(word, real_words, short_circuit=True):
    """get best spelling suggestion for word
    return on first match if short_circuit is true, otherwise collect all possible suggestions"""
    word = word.lower()
    if short_circuit:   # setting short_circuit makes the spellchecker much faster, but less accurate in some cases
        return ({word}                      & real_words or   #  caps     "inSIDE" => "inside"
                set(reductions(word))       & real_words or   #  repeats  "jjoobbb" => "job"
                set(vowelswaps(word))       & real_words or   #  vowels   "weke" => "wake"
                set(variants(word))         & real_words or   #  other    "nonster" => "monster"
                set(both(word))             & real_words or   #  both     "CUNsperrICY" => "conspiracy"
                set(double_variants(word))  & real_words or   #  other    "nmnster" => "manster"
                {"NO SUGGESTION"})
    else:
        return ({word}                      & real_words or
                (set(reductions(word))  | set(vowelswaps(word)) | set(variants(word)) | set(both(word)) | set(double_variants(word))) & real_words or
                {"NO SUGGESTION"})

def best(inputted_word, suggestions, word_model=None):
	"""choose the best suggestion in a list based on lowest hamming distance from original word, or based on frequency if word_model is provided"""

	suggestions = list(suggestions)

	def comparehamm(one, two):
		score1 = hamming_distance(inputted_word, one)
		score2 = hamming_distance(inputted_word, two)
		return cmp(score1, score2)  # lower is better

	def comparefreq(one, two):
		score1 = frequency(one, word_model)
		score2 = frequency(two, word_model)
		return cmp(score2, score1)  # higher is better

	freq_sorted = sorted(suggestions, cmp=comparefreq)[10:]     # take the top 10
	hamming_sorted = sorted(suggestions, cmp=comparehamm)[10:]  # take the top 10
	print 'FREQ', freq_sorted
	print 'HAM', hamming_sorted
	return ''

def spellchecker(word):
	# init the word frequency model with a simple list of all possible words
	word_model = train(file('/usr/share/dict/words').read())
	real_words = set(word_model)

	# add other texts here, they are used to train the word frequency model
	texts = [
	'sherlockholmes.txt',
	'lemmas.txt',
	'fox.txt',
	]
	# enhance the model with real bodies of english so we know which words are more common than others
	word_model = train_from_files(texts, word_model)
	while True:
		word1 = word
		word = recognizeString(5,word1)
		possibilities = suggestions(word, real_words, short_circuit=False)
		short_circuit_result = suggestions(word, real_words, short_circuit=True)
		if VERBOSE:
			print [(x, word_model[x]) for x in possibilities]
			print best(word, possibilities, word_model)
			print '---'
			a = [(x, word_model[x]) for x in short_circuit_result]
			name = ''
			maxi = 0
			for i in a:
				if i[1] > maxi:
					maxi = i[1]
					name = i[0]
			return name
		if VERBOSE:
			return best(word,short_circuit_result, word_model)


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
#t = turtle.Turtle()

#screen.addshape(image)
#turtle.shape(image)
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
f = open("oldwords.txt", 'w+')
g = open("newwords.txt", 'w+')
prev_word = ""
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
	cv2.putText(img, prev_word, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,0,255),2,cv2.LINE_AA)
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
	if done>=10 and start:
		print word
		f.write(word + "\n")
		rest = spellchecker(word)
		print "Word Detected: " + str(rest)
		g.write(rest + "\n")
		prev_word = rest
		word = ''
		done = 0
		start=False
		word = ""
	cv2.imwrite("keyboard.png", img)
	k=cv2.waitKey(33)
	if k == 32:
		f.close()
		g.close()
		break
	else:
		continue

	



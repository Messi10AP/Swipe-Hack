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
from turtle import *
screen = Screen()
s.bgpic("keyboard.png")

feed = cv2.VideoCapture(0)

while True:
	ret, i = feed.read()
	i.imwrite("k.png", i)
	s.bgpic("k.png")
	s.update()

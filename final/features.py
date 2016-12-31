import scipy.misc
from math import *
from numpy import *
import numpy as numpy
from glob import *
import csv
import cv2
from PIL import Image
from sys import*
import matplotlib.pyplot as plt
import pylab as plt


def get_edge_points(img):
	edges = []
	canny_image = cv2.Canny(img, 100, 200)

	for i in range(len(canny_image)):
		for j in range(len(canny_image[0])):
			if canny_image[i][j] == 255:
				edges.append((i,j))
	
	return edges

def get_solidity(img):
	#print "IMAGEEEE! get_solidity "
	area = cv2.contourArea(img)
	hull = cv2.convexHull(img)
	hull_area = cv2.contourArea(hull)
	if hull_area == 0:
		hull_area = 0.0001
	solidity = float(area)/hull_area

	return solidity

def get_corner_points(img, maxFeat):
	feature_params = dict( maxCorners = maxFeat, qualityLevel = 0.6, minDistance = 7, blockSize = 7 )
	corners = cv2.goodFeaturesToTrack(img, mask = None, **feature_params)
	return corners


def display_image(img):
	cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
	cv2.imshow('Image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

def get_moments(img):
	return cv2.moments(img)

def initialProcessingAndGetContours(path):
	img = cv2.imread(path) 
	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY_INV)
	se = ones((15,15), dtype='uint8')
	image_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se)
	mask = zeros(imgray.shape[:2], uint8)
	image,contours, hierarchy = cv2.findContours(image_close,1,2)
	cv2.drawContours(mask,contours,-1,(200,200,0),3)

	return mask

def get_image_area(image):
	nonzero = count_nonzero(image)
	return nonzero


files = glob('/home/roliveira/Documents/Sorbus aucuparia/*.tif')
images = []	
i =0 
for f in files:
	
	#image = initialProcessingAndGetContours(f)
	img = cv2.imread(f) 

	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(imgray,(5,5),0)

	ret,thresh = cv2.threshold(blur,0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	
	se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
	se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

	#image_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se1)
	imagef = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, se2)

	print 'sorbus/'+str(i)+'.jpg' 
	cv2.imwrite('sorbus/'+str(i)+'.jpg' ,imagef)
	i=i+1
	cv2.destroyAllWindows()

	#display_image(imagef)

	#corners = get_corner_points(image,100)
	#moments = get_moments(image)
	#print moments
	#print corners 
	#display_image(image)
	#print get_image_area(image)
	#print len(get_edge_points(image))
	#display_image(image)
	#print get_solidity(image)
	#break; 
	


	#break; 
	

'''
File name: Utils.py
Objective: utility tools for leave classifier
Author: Andy D. Martinez & Daniela Florit
Date created: 11/06/2016
Python Version: 2.7.12
'''
import scipy.misc
from math import *
from numpy import *
import numpy as numpy
from Data import *
from glob import *
import cv2
from PIL import Image
from sys import*
import pylab as plt

#---------------------------------------Variables----------------------------------------
test_kaggle_table = 'Data/Dataset1/data_binary_Kaggle/test.csv'

train_kaggle_table = 'Data/Dataset1/data_binary_Kaggle/nosso_train.csv'

kaggle_images_path = 'Data/Dataset1/data_binary_Kaggle' 

label_to_number = {'Populus_Nigra': 69, 'Acer_platanoides':222, 'Sorbus aucuparia': 1211, 'Quercus_Rubra': 7, 'Ginkgo_Biloba': 51, 'Acer_Capillipes' : 2,\
					'Tilia_Tomentosa': 3, 'Fagus_Sylvatica': 15, 'Olea_Europaea':1, 'Quercus_Shumardii':11, \
 					'Prunus_Avium': 43, 'Ilex_Aquifolium': 61, 'Castanea_Sativa': 44, 'Acer_Circinatum' : 12, \
					'Acer_Platanoids': 24, 'Acer_Palmatum': 17, 'Liquidambar_Styraciflua': 52, 'Quercus_Vulcanica':122}



#-------------------------------------Image Processing Tools-----------------------------
def save_image(arr, name):
	scipy.misc.imsave(name, arr)	

def get_cnt(img):
	ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	img,contours,hierarchy = cv2.findContours(thresh, 1, 2)

	cnt = contours[0]
	return cnt

def get_moments(img):
	return cv2.moments(get_cnt(img))

def get_solidity(img):
	#print "IMAGEEEE! get_solidity "
	#print img
	cnt = get_cnt(img)
	area = cv2.contourArea(cnt)
	hull = cv2.convexHull(cnt)
	hull_area = cv2.contourArea(hull)
	if hull_area == 0:
		hull_area = 0.0001
	solidity = float(area)/hull_area

	return solidity


def max_x_diff(img):
	edge_points = get_edge_points(img)

	min_x = maxint
	max_x = -1

	for point in edge_points:
		x = point[0]
		
		if x > max_x:
			max_x = x
		if x < min_x:
			min_x = x

	x_diff = 1.0*max_x - min_x
	return x_diff

def max_y_diff(img):
	edge_points = get_edge_points(img)

	min_y = maxint
	max_y = -1

	for point in edge_points:
		y = point[1]
		
		if y > max_y:
			max_y = y
		if y < min_y:
			min_y = y

	y_diff = 1.0*max_y - min_y
	return y_diff



def display_image(img):
	cv2.imshow('Image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

def get_edge_points(img):
	edges = []
	canny_image = cv2.Canny(img, 100, 200)

	for i in range(len(canny_image)):
		for j in range(len(canny_image[0])):
			if canny_image[i][j] == 255:
				edges.append((i,j))
	
	return edges

def get_corner_points(img, maxFeat):

	print 'Extracting features in new image...'
	#print maxFeat

	feature_params = dict( maxCorners = maxFeat, qualityLevel = 0.6, minDistance = 7, blockSize = 7 )
	corners = cv2.goodFeaturesToTrack(img, mask = None, **feature_params)
	return corners
	
	'''
	for point in corners:
		x, y = point.ravel()
	'''

#make sure that img was read as a grayscale image
def get_binary_image(img):


	#img = cv2.medianBlur(img,5)
	#ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	return img #cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	

def get_binary_image_contours(imgray):
	#image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 0)    


	ret,thresh = cv2.threshold(imgray,127,255,1)

	image,contours, hierarchy = cv2.findContours(thresh,1,2)

	#print len(image)
	#contours=contours[1::]
	mask = zeros(imgray.shape[:2], uint8)

	cv2.drawContours(mask,contours,-1,(255,0,0),3)



	# Perform morphology
	#se = ones((4,4), dtype='uint8')
	#image_close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)



	#print "MASK"
	#print mask

	
	return mask


def get_image_area(image):
	nonzero = count_nonzero(image)
	return nonzero

#-------------------------------------File System Processing Tools-----------------------

#headers is a list or numpy array
#rows is a list of lists
#file_name does not contain the .csv extension
def build_excel_file(rows, file_name, headers=None):
	formated_string = ''

	if headers != None:
		for h in headers:
			formated_string = formated_string + str(h)+','

		formated_string = formated_string[0:len(formated_string)-1]
		formated_string+='\n'

	for row in rows:
		for e in row:
			formated_string += str(e)+','
		formated_string = formated_string[0:len(formated_string)-1]
		formated_string+='\n'

	f = open(file_name+'.csv', 'w')
	f.write(formated_string)
	f.close()

	
	 

#table_path is the path to the excel_table
#return a Data object with feature vectors and labels included
def read_excel_table(table_path):
	#feature_names = []
	#feature_vectors = []
	#data = Data()
	headers = []
	rows = []
	
	f = open(table_path, 'r')
	
	for line in f:
		line = line.replace('\n', '')
		splited_line = line.split(',')
		splited_line = filter(lambda a: a != '', splited_line)
		splited_line = filter(lambda a: a != '\r', splited_line)
	
		if len(headers) == 0:
			headers = splited_line
		else:
			#feature_vectors.append([float(feature) for feature in splited_line])
			rows.append(splited_line)
	
	f.close()
	
	#data.set_feature_vectors(array(feature_vectors))
	#data.set_feature_names(array(feature_names))
	
	return (headers, rows)

def read_kaggle_test_table(table_path = test_kaggle_table):
	data = Data()
	feature_vectors = []
	ids = []
	(feature_names, feature_vectors_str) = read_excel_table(table_path)

	feature_names = feature_names[1:len(feature_names)]
	
	for row in feature_vectors_str:
		ids = ids+row[0:1]
		row = row[1:len(row)]
		feature_vectors.append([float(feature) for feature in row])
	
	data.set_feature_vectors(array(feature_vectors))
	data.set_feature_names(array(feature_names))
	data.set_table_ids(array(ids))
	
	return data

#Remember that Column 2 contains the classification of the feature vector
def read_kaggle_training_table(table_path = train_kaggle_table, to_number=None):
	data = Data()
	feature_vectors = []
	labels = []
	ids = []
	numeric_labels = []
	(feature_names, feature_vectors_str) = read_excel_table(table_path)

	feature_names = feature_names[2:len(feature_names)]


	for row in feature_vectors_str:
		labels.append(row[1])
		if to_number == None:
			numeric_labels.append(label_to_number[row[1]])
			#print "entrei if"
		else:
			numeric_labels.append(to_number[row[1]])
			#print "entrei elese"


		#print numeric_labels

		ids.append(row[0])
		row = row[2:len(row)]
		
		feature_vectors.append([float(feature) for feature in row])
	
	data.set_feature_vectors(array(feature_vectors))
	data.set_feature_names(array(feature_names))
	data.set_labels(array(labels))
	data.set_table_ids(array(ids))
	data.set_numeric_labels(array(numeric_labels))	


	return data

#image_path is the path to the image
#returns the image as a 2D numpy array (Grayscale)
def read_image_grayscale(image_path):
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	return image
	

#returns the image as a 3D numpy array (RGB)
def read_image_color(image_path):
	image = cv2.imread(image_path)
	
	return image

#This function reads all kaggle leaves images in grayscale on the provided path
#It returns a data object that contains images, ids, labels. If no label
#in the Kaggle train table (which means they belong to Kaggle testing set)
#their label will be None
def read_all_kaggle_gray_scale_images(images_directory_path = kaggle_images_path):
	files = glob(images_directory_path+'/*.jpg')
	images = []	
	ids = []
	labels = []
	numeric_labels = []
	data = Data()
		
	
	#in order to get the labels
	train_table_data = read_kaggle_training_table(train_kaggle_table)

	for f in files:
		splited_file_path = f.split('/')
		file_name = splited_file_path[len(splited_file_path)-1]
		splited_file_name = file_name.split('.')

		ids.append(splited_file_name[0])
		images.append(read_image_grayscale(f))
	
		labels.append(None)
		numeric_labels.append(None)

	table_ids = train_table_data.get_table_ids()
	table_labels = train_table_data.get_labels()

	for i in range(len(ids)):
		image_id = ids[i]		

		for j in range(len(table_ids)):
			if table_ids[j] == image_id:
				labels[i] = table_labels[j]
				numeric_labels[i] = label_to_number[table_labels[j]]
				break
		

	data.set_images_binary(array(images))
	data.set_table_ids(array(ids))
	data.set_labels(array(labels))
	data.set_numeric_labels(array(numeric_labels))

	return data

#This function reads all leaves images in rgb on the provided path
#It returns a data object with colored images setted
def read_all_color_images(images_directory_path):
	files = glob(images_directory_path+'/*.jpg')
	images = []	
	data = Data()

	for f in files:
		images.append(read_image_color(f))
	

	data.set_images_color(array(images))
	return data

#This function reads all leaves images in rgb on the provided path
#It returns a data object with colored images setted
def read_all_grayscale_images(images_directory_path):
	files = glob(images_directory_path+'/*.jpg')
	images = []	
	data = Data()
	ids = []

	for f in files:
		images.append(read_image_grayscale(f))
		ids.append(f)

	data.set_images_binary(array(images))
	data.set_table_ids(array(ids))
	return data
	

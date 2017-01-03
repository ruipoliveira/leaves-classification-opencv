from math import *
from numpy import *
import numpy as numpy
from Data import *
from glob import *
import cv2
from sys import*

#---------------------------------------Variables----------------------------------------

label_to_number = {'Populus_Nigra': 1, 'Acer_platanoides':2, 'Sorbus aucuparia': 3, 'Quercus_Rubra': 4, 
					'Ginkgo_Biloba': 5, 'Acer_Capillipes' : 6,'Tilia_Tomentosa': 7, 'Fagus_Sylvatica': 8, 
					'Olea_Europaea':9, 'Quercus_Shumardii':10, 'Prunus_Avium': 11, 'Ilex_Aquifolium': 12, 
					'Castanea_Sativa': 13, 'Acer_Circinatum' : 14, 'Acer_Platanoids': 15, 'Acer_Palmatum': 16,
					'Liquidambar_Styraciflua': 17, 'Quercus_Vulcanica':18}

#-------------------------------------Image Processing Tools-----------------------------

def get_cnt(img):
	ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	img,contours,hierarchy = cv2.findContours(thresh, 1, 2)

	cnt = contours[0]
	return cnt

def get_moments(img):
	return cv2.moments(get_cnt(img))

def get_solidity(img):

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
	cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
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

	feature_params = dict( maxCorners = maxFeat, qualityLevel = 0.6, minDistance = 7, blockSize = 7 )
	corners = cv2.goodFeaturesToTrack(img, mask = None, **feature_params)
	return corners


def get_binary_image_contours(imgray):

	ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY_INV)
	#se = ones((15,15), dtype='uint8')
	#image_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se)
	mask = zeros(imgray.shape[:2], uint8)

	image,contours, hierarchy = cv2.findContours(thresh,1,2)

	cv2.drawContours(mask,contours,-1,(200,200,0),3)

	#display_image(mask)
	return mask


def get_image_area(image):
	nonzero = count_nonzero(image)
	return nonzero

def read_csv_table(table_path):

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
			rows.append(splited_line)
	f.close()
	
	return (headers, rows)



def read_training_table(table_path, to_number=None):
	data = Data()
	feature_vectors = []
	labels = []
	ids = []
	numeric_labels = []
	(feature_names, feature_vectors_str) = read_csv_table(table_path)

	feature_names = feature_names[2:len(feature_names)]


	for row in feature_vectors_str:
		labels.append(row[1])
		if to_number == None:
			numeric_labels.append(label_to_number[row[1]])
			#print "entrei if"
		else:
			numeric_labels.append(to_number[row[1]])

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


def read_image_grayscale(image_path):
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return image
	

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
	

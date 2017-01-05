import sys
import os
import glob
from Data import *
from Classifiers import *
from Utils import *
from FeatureExtractors import *
from PyQt4 import QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *

all_features = 'all'
no_moments_features = 'nm'
feature_extractors = [all_features, no_moments_features]

#Variables
dynamic_input_dir = 'Test_Leaves'
training_tables = 'Data/Training_Tables/'
output_all = training_tables + 'output_all.csv'
output_nm = training_tables + 'output_nm.csv'

#Default Values
default_feature_extractor = no_moments_features

def read_user_input():
	feature_extractor = None

	if len(sys.argv) == 1:
		feature_extractor = default_feature_extractor
	else:
		feature_extractor = sys.argv[1]
	
	return feature_extractor


def get_test_data(feature_extractor):


	test_data = read_all_grayscale_images(dynamic_input_dir)		

	images = test_data.get_images_binary()
	binary_images = []
	feature_vecs = []
	f_e = FeatureExtractors()
	feature_names = None

	for im in images:
		print 'Extracting features in image '

		b_im = get_binary_image_contours(im)
		#display_image(b_im)
		
		binary_images.append(b_im)
		
		if feature_extractor == all_features:

			(feature_names, features) = f_e.all_feature_extractor(b_im)
			feature_vecs.append(features)
		elif feature_extractor == no_moments_features:
			(feature_names, features) = f_e.no_moments_feature_extractor(b_im)
			feature_vecs.append(features)	
		
	test_data.set_feature_vectors(array(feature_vecs))
	test_data.set_images_binary(array(binary_images))
	test_data.set_feature_names(array(feature_names))
	
	return test_data
		
			

def get_data_and_classify(classifier, feature_extractor):
	data = None
	train_data = None
	test_data = None
	
	if feature_extractor == all_features:
		data = read_training_table(output_all)
		
	elif feature_extractor == no_moments_features:
		data = read_training_table(output_nm)

	train_data = data

	test_data = get_test_data(feature_extractor) 
	
	classifier.set_training_data(train_data)
	classifier.set_testing_data(test_data)
	
	classifier.train()
	classifier.predict()
	
	return test_data


def display_results(prediction_data):
	predictions = prediction_data.get_predictions()
	ids = prediction_data.get_table_ids()
	S = None
	
	print '+++++++ Previsoes' + dynamic_input_dir + ' +++++++ '
	for i in range((len(ids)/2)-1):
		print str(ids[i]) + ': ' + str(predictions[i])
	

def display_image(prediction_data):
	predictions = prediction_data.get_predictions()
	ids = prediction_data.get_table_ids()
	S = None
	
	app = QtGui.QApplication(sys.argv)
	widget = QtGui.QWidget()
	layout = QtGui.QGridLayout()

	myFont=QtGui.QFont()
	myFont.setBold(True)
	l1 = QLabel()
	l1.setText("Input image")
	l1.setFont(myFont)
	l2 = QLabel()
	l2.setText("Result (sample image)")
	l2.setFont(myFont)

	layout.addWidget(l1, 0, 0)
	layout.addWidget(l2, 0, 1)

	window = QtGui.QMainWindow()
	window.setGeometry(0, 0, 40, 20)
		
	for i in range(len(ids)):
		print str(ids[i]) + ': ' + str(predictions[i])
			
		pic1 = QtGui.QLabel(window)
		pic1.setGeometry(1, 1, 40, 10)
		pixmap1 = QtGui.QPixmap(os.getcwd() + str("/")+str(ids[i]))
		pixmap21 = pixmap1.scaled(164, 164)
		pic1.setPixmap(pixmap21)
		layout.addWidget(pic1, i+1, 0)

		pic2 = QtGui.QLabel(window)
		pic2.setGeometry(1, 1, 40, 10)

		folder = os.getcwd() + str("/Data/DataTraining/") + str(predictions[i])
		allfiles = os.listdir(folder)
		pathfirst = folder+ str("/")+str(allfiles[0])
		pixmap2 = QtGui.QPixmap(pathfirst)

		#pixmap = QtGui.QPixmap(folder+ str("/")+str(allfiles[0]))
	
		pixmap22 = pixmap2.scaled(164, 164)
		pic2.setPixmap(pixmap22)
		layout.addWidget(pic2, i+1, 1)
	
	widget.setWindowTitle("Leaves classification, Computer Vision 2016")
	widget.setLayout(layout)
	widget.show()
	sys.exit(app.exec_())


def display_input_prameters(feature_extractor):
	os.system('clear')
	print '++++++++++++++++++++++++Parameters+++++++++++++++++++++++'
	if feature_extractor == no_moments_features: 
		print 'Extracting Features without momments.'
	else: 
		print 'Extracting Features with moments.'
		
	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++'


#----------------------------------Main--------------------------------

feature_extractor = read_user_input()

display_input_prameters(feature_extractor)

classifier = SVC_Classifier()

prediction_data = get_data_and_classify(classifier, feature_extractor)

display_results(prediction_data)
display_image(prediction_data)



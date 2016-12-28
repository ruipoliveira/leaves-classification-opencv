import sys

from Data import *
from Classifiers import *
from Utils import *
from FeatureExtractors import *

#Expected Input: python Leaves_Classifier_final.py mode classification_type classifier feature_extractor 



#feature_extractor
all_features = 'all'
no_moments_features = 'nm'
feature_extractors = [all_features, no_moments_features]

#Variables
dynamic_input_dir = 'Data/Liquidambar_Styraciflua'
training_tables = 'Data/Training_Tables'
full_table = training_tables + '/nossa_full_table.csv'
partial_table = training_tables + '/nossa_partial_table.csv'


#Default Values
default_feature_extractor = no_moments_features


def read_user_input():
	mode = 'd'
	classification_type = 's'
	classifier = 'svm'
	feature_extractor = None
	
	#Set default values
	if len(sys.argv) == 1:
		feature_extractor = default_feature_extractor
	else:
		feature_extractor = sys.argv[1]
	
	return (mode, classification_type, classifier, feature_extractor)


def get_test_data(feature_extractor):
	test_data = read_all_grayscale_images(dynamic_input_dir)		
	#print test_data

	images = test_data.get_images_binary()
	#print images
	binary_images = []
	feature_vecs = []
	f_e = Feature_Extractors()
	feature_names = None
	#m = 0
	for im in images:

		b_im = get_binary_image_contours(im)
		
		#display_image(b_im)
		#print b_im 
		#print "*******************"

		binary_images.append(b_im)
		
		#save_image(b_im, str(m)+'.jpg')#delete this
		#m+=1
		if feature_extractor == all_features:
			(feature_names, features) = f_e.all_feature_extractor(b_im)
			print "uma"
			print feature_names
			print  features

		elif feature_extractor == no_moments_features:
			(feature_names, features) = f_e.all_five_feature_extractor(b_im)
	




	
	return (feature_names, features) 
		
			

def get_data(feature_extractor):
	data = None
	train_data = None
	test_data = None
	
	print "asdsad"

	(feature_names, features) = get_test_data(feature_extractor) 

	print feature_names
	print features 
	
	return (train_data, test_data)




		
def display_input_prameters(mode, classification_type, classifier, feature_extractor):
	print '++++++++++++++++++++++++Parameters+++++++++++++++++++++++'
	print 'Mode: ' + mode
	print 'Classification Type: ' + classification_type
	print 'Classifier: ' + classifier
	print 'Feature Extractor: ' + feature_extractor
	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
#----------------------------------Main--------------------------------
(mode, classification_type, classifier, feature_extractor) = read_user_input()

display_input_prameters(mode, classification_type, classifier, feature_extractor)


(train_data, test_data) = get_data( feature_extractor)

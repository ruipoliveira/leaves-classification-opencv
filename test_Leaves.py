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
dynamic_input_dir = 'Test_Leves'
training_tables = 'Data/Training_Tables'
full_table = "output.csv" #training_tables + '/nossa_full_table.csv'
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

def get_classifier(which_classifier):
	classifier = SVC_Classifier()
		
	return classifier

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
		
		display_image(b_im)
		#print b_im 
		#print "*******************"

		binary_images.append(b_im)
		
		#save_image(b_im, str(m)+'.jpg')#delete this
		#m+=1
		if feature_extractor == all_features:
			(feature_names, features) = f_e.all_feature_extractor(b_im)
			feature_vecs.append(features)
		elif feature_extractor == no_moments_features:
			(feature_names, features) = f_e.all_five_feature_extractor(b_im)
			feature_vecs.append(features)	
	
	test_data.set_feature_vectors(array(feature_vecs))
	test_data.set_images_binary(array(binary_images))
	test_data.set_feature_names(array(feature_names))
	
	return test_data
		
			

def get_data(feature_extractor):
	data = None
	train_data = None
	test_data = None
	
	print "asdsad"
	
	if feature_extractor == all_features:
		#do all features
		data = read_kaggle_training_table(full_table)
		#print data 
		
	elif feature_extractor == no_moments_features:
		#do no moments features
		data = read_kaggle_training_table(partial_table)
		#print data 

	train_data = data
	test_data = get_test_data(feature_extractor) 

	print train_data
	
	return (train_data, test_data)


def classify(classifier, train_data, test_data):
	

	classifier.set_training_data(train_data)
	classifier.set_testing_data(test_data)
	
	classifier.train()
	classifier.predict()
	
	return test_data


def display_results(prediction_data, mode, classification_type):
	predictions = prediction_data.get_predictions()
	ids = prediction_data.get_table_ids()
	S = None
	

	print '------Predictions for images inside ' + dynamic_input_dir + '--------------'
	for i in range(len(ids)):
		print str(ids[i]) + ': ' + str(predictions[i])
	
		
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

classifier = get_classifier(classifier)

(train_data, test_data) = get_data( feature_extractor)

prediction_data = classify(classifier, train_data, test_data)

print prediction_data

display_results(prediction_data, mode, classification_type)	
#display_image(prediction_data.get_images_binary()[0])

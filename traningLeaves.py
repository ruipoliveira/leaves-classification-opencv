import sys
import os
from Data import *
from Classifiers import *
from Utils import *
from FeatureExtractors import *
import csv

#Expected Input: python Leaves_Classifier_final.py mode classification_type classifier feature_extractor 



#feature_extractor
all_features = 'all'
no_moments_features = 'nm'
feature_extractors = [all_features, no_moments_features]

#Variables
dynamic_input_dir = 'Data/DataTraining/'
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


def get_test_data(feature_extractor, dynamic_input_dir):

	alldir = [x[0] for x in os.walk(dynamic_input_dir)][1:]
	id = 1
	first = True 
	with open("output.csv","w") as f:
		wr = csv.writer(f,delimiter=",")

		for input_dir in alldir: 

			print input_dir
			test_data = read_all_grayscale_images(input_dir)		
			#print test_data

			images = test_data.get_images_binary()
			#print images
			binary_images = []
			feature_vecs = []
			f_e = Feature_Extractors()
			feature_names = None
			
			label = input_dir.split("/")[2]

			for im in images:

				b_im = get_binary_image_contours(im)
				
				#display_image(b_im)
				#print b_im 
				#print "*******************"

				binary_images.append(b_im)
				
				#save_image(b_im, str(m)+'.jpg')#delete this
				#m+=1
				if feature_extractor == all_features:
					(feature_names, features) = f_e.all_feature_extractor_traning(b_im, id, label)
					id = id + 1 
					if first == True : 
						wr.writerow(feature_names)
						first = False; 

					wr.writerow(features)

				elif feature_extractor == no_moments_features:
					(feature_names, features) = f_e.all_five_feature_extractor(b_im)
		

	return (feature_names, features) 
		





			



		
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

get_test_data(feature_extractor, dynamic_input_dir)

	#if (cenas == True):  
	#	print "ignorei o primeiro :3"
#		cenas = False 
#	else : 	
#		(train_data, test_data) = get_data( feature_extractor, input_dir)

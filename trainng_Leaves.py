import sys
import os
from Data import *
from Classifiers import *
from Utils import *
from FeatureExtractors import *
import csv

all_features = 'all'
without_moments_features = 'nm'
feature_extractors = [all_features, without_moments_features]

#Variables
dynamic_input_dir = 'Data/DataTraining/'

#Default Values
default_feature_extractor = without_moments_features

def read_user_input():
	feature_extractor = None
	
	if len(sys.argv) == 1:
		feature_extractor = default_feature_extractor
	else:
		feature_extractor = sys.argv[1]
	
	return feature_extractor


def get_test_data(feature_extractor, dynamic_input_dir):

	alldir = [x[0] for x in os.walk(dynamic_input_dir)][1:]
	id = 1
	first = True 
	with open("output_"+feature_extractor+".csv","w") as f:
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
				binary_images.append(b_im)

				if feature_extractor == all_features:
					(feature_names, features) = f_e.all_feature_extractor_traning(b_im, id, label)
					id = id + 1 
					if first == True : 
						wr.writerow(feature_names)
						first = False; 

					wr.writerow(features)

				elif feature_extractor == without_moments_features:
					print "without moments"
					(feature_names, features) = f_e.all_five_feature_extractor_traning(b_im,id, label)
					id = id + 1 
					if first == True : 
						wr.writerow(feature_names)
						first = False;

					wr.writerow(features)

	return (feature_names, features) 
		

def display_input_prameters(feature_extractor):
	print(chr(27) + "[2J")
	print '++++++++++++++++++++++++Parameters+++++++++++++++++++++++'
	print 'Feature Extractor: ' + feature_extractor
	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++'


feature_extractor =  read_user_input()

display_input_prameters(feature_extractor)

get_test_data(feature_extractor, dynamic_input_dir)



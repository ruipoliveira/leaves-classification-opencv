from random import *

class DataStructure:
	def __init__(self):
		self.length = 0
		self.feature_vectors = []
		self.labels = []
		self.predictions = []
		self.images_binary = []
		self.images_color = []
		self.feature_names = []
		self.table_ids = []
		self.numeric_labels = []
	
	#feature_vectors = [phi(I1), phi(I2),...,phi(In)]
	def set_feature_vectors(self, feature_vectors):
		self.length = len(feature_vectors)
		self.feature_vectors = feature_vectors
	
	def get_feature_vectors(self):
		return self.feature_vectors
	
	
	def set_labels(self, labels):
		self.length = len(labels)
		self.labels = labels
	
	def get_labels(self):
		return self.labels
		
	
	def set_numeric_labels(self, numeric_labels):
		self.length = len(numeric_labels)
		self.numeric_labels = numeric_labels
	
	def get_numeric_labels(self):
		return self.numeric_labels
		

	def set_predictions(self, predictions):
		self.length = len(predictions)
		self.predictions = predictions	
	
	def get_predictions(self):
		return self.predictions
	
	def set_images_binary(self, images_binary):
		self.length = len(images_binary)
		self.images_binary = images_binary
	
	def get_images_binary(self):
		return self.images_binary


	def set_images_color(self, images_color):
		self.length =len(images_color)
		self.images_color = images_color
	
	def get_images_color(self):
		return self.images_color
	
	def set_feature_names(self, feature_names):
		self.length = len(feature_names)
		self.feature_names = feature_names
	
	def get_feature_names(self):
		return self.feature_names
		
	def set_table_ids(self, table_ids):
		self.length = len(table_ids)
		self.table_ids = table_ids
	
	def get_table_ids(self):
		return self.table_ids

	def get_length(self):
		return self.length
	
	def __str__(self):
		string = ''
		string += 'length: ' + str(self.length) + '\n \n'
		string += 'table_ids: ' + str(self.table_ids) + '\n \n'
		string += 'feature_names: ' + str(self.feature_names) + '\n \n'
		string += 'feature_vectors: ' + str(self.feature_vectors) + '\n \n'
		string += 'labels: ' + str(self.labels) + '\n \n'
		string += 'predictions: ' + str(self.predictions) + '\n \n'
		string += 'images_binary: ' + str(self.images_binary) + '\n \n'
		string += 'images_color: ' + str(self.images_color) + '\n \n'
		
		return string


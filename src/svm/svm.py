from sklearn import svm
from sklearn.cross_validation import KFold
import numpy as np
import random
import csv

# Add this project's src folder to the path
import sys
import os
sys.path.append(os.path.abspath('../'))

import data_processing as dp


class SVMRunner (object):
	'''
	SVM classifer that trains on examples of the form:

		('id', {'word1': 2, 'word2':1, 'word3':1, ...}, 'class_name')

	'''

	# scikit-learn SVM classifier variants
	REPRESENTATIONS = [
		'SVC',
		'NuSVC',
		'LinearSVC'
	]


	def __init__ (self, representation='LinearSVC'):
		self.features = []
		self.classifier = getattr(svm, representation)()


	def get_classifier (self):
		return self.classifier


	def extract_features (self, examples):
		'''
		Extract set of features/words from examples.
		'''

		features = set([ word for e in examples for word in e[1] ])
		self.features = list(features)


	def example_to_vector (self, example):
		'''
		Package example word counts into a vector.
		'''

		word_dict = example[1]
		feature_vals = [ (float(word_dict[f]) if f in word_dict else 0.0) for f in self.features ]
		return feature_vals


	def get_X_and_y (self, examples):
		'''
		Extract matrix of feature values and class labels from examples.
		'''

		# Matrix of feature values for examples
		X = []
		# Vector of corresponding class labels
		y = []
		for example in examples:
			X.append(self.example_to_vector(example))
			class_name = example[2]
			y.append(class_name)

		return (np.array(X), np.array(y))


	def train (self, examples):
		'''
		Train SVM classifier on set of examples.
		'''

		self.extract_features(examples)

		(X, y) = self.get_X_and_y(examples)
		self.classifier.fit(X, y)


	def classify (self, example):
		'''
		Takes in an example (which is missing the class), and outputs the
		predicted class according to the SVM classifier.
		'''

		return self.classifier.predict(self.example_to_vector(example))[0]


	def test (self, examples):
		'''
		Takes a list of labelled examples, outputs the overall accuracy of
		the classifier on the list. (Classifier must be trained before
		calling this function.)
		'''
		(X, y) = self.get_X_and_y(examples)
		return self.classifier.score(X, y)


	def pred (self, examples):
		'''
		Predicts on a list of examples.
		'''

		preds = []

		for example in examples:
			e_Id = example[0]
			preds.append([e_Id, self.classify(example)])
		
		return preds


class CrossValTester(object):
	'''
	Given a data set, allows one to perform cross validation with the 
	SVM classifier.

	The dataset must be a list of examples, where each example has the form

		(<id>, <feature_counts>, <class_name>)

	id should be a string, feature_counts should be a dictionary with 
	numeric values, and class_name should be a string.

	So the dataset should look something like this:

		[
			...
			(1, {'feature1':3, 'feature2':1, ...}, 'physics'),
			...
		]
	'''


	def __init__ (self, dataset, representation='LinearSVC', limit=None):
		self.representation = representation

		self.dataset = dataset

		# Randomize the examples' ordering.
		random.shuffle(self.dataset)

		# Optionally limit data for improved time efficiency
		if limit is not None:
			self.dataset = self.dataset[:limit]

		self.size = len(self.dataset)


	def cross_validate (self, k=None):
		'''
		Divide the dataset set into k equal folds (If k doesn't divide the
		number of examples evenly, then the folds won't all be equal). For
		each 'fold' of cross validation, train a SVM classifier on all the
		data outside the fold, then test it on the data inside the
		fold, and repeat for all folds.  Keep a running tally of the number
		of classifications correct.
		'''


		# If k is not specified, do leave-one-out cross validation
		if k is None:
			k = self.size
		k = int(k)

		accuracy = 0.0

		fold = 1
		kf = KFold(self.size, n_folds=k, indices=False)
		for train_indices, test_indices in kf:
		    print 'Fold %i' % fold

		    train_set = (np.array(self.dataset))[train_indices]
		    test_set = (np.array(self.dataset))[test_indices]

		    classifier = SVMRunner(representation=self.representation)
		    classifier.train(train_set.tolist())

		    accuracy += classifier.test(test_set.tolist()) / k

		    fold += 1

		print 'OVERALL ACCURACY: %f' % accuracy

		# Return the overall accuracy 
		return accuracy


	def cross_validate_to_file (self, csv_filepath, k=None):
		'''
		Same as cross_validate except predictions on the folds are also output
		to a .csv file.
		'''

		# If k is not specified, do leave-one-out cross validation
		if k is None:
			k = self.size
		k = int(k)

		preds = []

		accuracy = 0.0

		fold = 1
		kf = KFold(self.size, n_folds=k, indices=False)
		for train_indices, test_indices in kf:
		    print 'Fold %i' % fold

		    train_set = (np.array(self.dataset))[train_indices]
		    test_set = (np.array(self.dataset))[test_indices]

		    classifier = SVMRunner(representation=self.representation)
		    classifier.train(train_set.tolist())

		    preds += classifier.pred(test_set.tolist())
		    accuracy += classifier.test(test_set.tolist()) / k

		    fold += 1

		with open(csv_filepath, 'wb') as csv_file:
			writer = csv.writer(
				csv_file,
				delimiter=',',
				quotechar='"',
				quoting=csv.QUOTE_ALL
			)

			writer.writerow(['id', 'category'])

			for pred in preds:
				writer.writerow(pred)

		return accuracy


class CrossValCase(object):
	'''
	Runner for cross-validation.
	'''

	# Permissable representations of the dataset.
	ALLOWED_REPRESENTATIONS = [ 
		'as_tficf',
		'as_modified_tficf',
		'as_trimmed_tficf',
		'as_frequencies',
		'as_tfidf'
	]

	# Number of folds.
	K = 10


	def __init__(self):
		pass


	def run(
		self,
		representation,
		lemmatize,
		find_specials,
		remove_stops,
		use_digrams,
		classifier_rep,
		limit=None
	):
		'''
		Generate dataset and apply cross-validation.
		'''
		
		assert(representation in self.ALLOWED_REPRESENTATIONS)
		data_manager = dp.Data(limit=limit)

		# This requests the data manager to calculate the representation
		# specified by the parameters passed to the run method
		dataset = getattr(data_manager, representation)(
			use_cache=True,
			lemmatize=lemmatize, 
			find_specials=find_specials,
			remove_stops=remove_stops,
			use_digrams=use_digrams,
			data_part='train',
			as_vect=False
		)

		cross_val_tester = CrossValTester(
			dataset=dataset,
			representation=classifier_rep
		)

		accuracy = cross_val_tester.cross_validate(self.K)

		return accuracy


class GenPredictionsCase (object):
	'''
	Runner for generating predictions on the training (through cross-validation)
	and test set.
	'''


	# Permissable representations of the dataset.
	ALLOWED_REPRESENTATIONS = [ 
		'as_tficf',
		'as_modified_tficf',
		'as_trimmed_tficf',
		'as_frequencies',
		'as_tfidf'
	]

	# Number of folds in cross-validation
	K = 10


	def __init__(self):
		pass


	def run(
		self,
		representation,
		lemmatize,
		find_specials,
		remove_stops,
		use_digrams,
		classifier_rep,
		limit=None
	):
		'''
		Generate training and test sets and then generate predictions.
		'''
		
		assert(representation in self.ALLOWED_REPRESENTATIONS)
		data_manager = dp.Data(limit=limit)

		# This requests the data manager to calculate the representation
		# specified by the parameters passed to the run method
		train_set = getattr(data_manager, representation)(
			use_cache=True,
			lemmatize=lemmatize, 
			find_specials=find_specials,
			remove_stops=remove_stops,
			use_digrams=use_digrams,
			data_part='train',
			as_vect=False
		)

		print 'CROSS VALIDATION'

		cross_val_tester = CrossValTester(
			dataset=train_set,
			representation=classifier_rep
		)
		
		train_csv_filepath = classifier_rep + '_' + representation + '_train_preds.csv'
		accuracy = cross_val_tester.cross_validate_to_file(
			k=self.K,
			csv_filepath=train_csv_filepath
		)

		print 'OVERALL ACCURACY:', accuracy

		print 'TRAIN'

		classifier = SVMRunner(representation=classifier_rep)
		classifier.train(train_set)

		print 'PREDICT'

		data_manager = dp.Data(limit=None)
		test_set = getattr(data_manager, representation)(
			use_cache=True,
			lemmatize=lemmatize, 
			find_specials=find_specials,
			remove_stops=remove_stops,
			use_digrams=use_digrams,
			data_part='test',
			as_vect=False
		)
		
		print 'Getting predictions on test set...'
		test_preds = classifier.pred(test_set)

		print 'Writing predictions to file...'
		test_csv_filepath = classifier_rep + '_' + representation + '_test_preds.csv'
		with open(test_csv_filepath, 'wb') as csv_file:
			writer = csv.writer(
				csv_file,
				delimiter=',',
				quotechar='"',
				quoting=csv.QUOTE_ALL
			)

			writer.writerow(['id', 'category'])

			for pred in test_preds:
				writer.writerow(pred)

		# To make it compatible with sim_sweep
		# return 0.0

		return accuracy


if __name__ == '__main__':
	limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

	CrossValCase().run(
		representation='as_tficf',
		lemmatize=True,
		find_specials=True,
		remove_stops=True,
		use_digrams=False,
		limit=100,
		classifier_rep='LinearSVC'
	)

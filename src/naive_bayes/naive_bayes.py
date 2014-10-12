import numpy as np
import json
import random
import copy
from collections import Counter, defaultdict

# add this project's src folder to the path
import sys
import os
sys.path.append(os.path.abspath('../'))

import data_processing as dp


def read_dataset(fname):
	data = json.loads(open(fname, 'r').read())
	data = [tuple(r) for r in data]
	return data

class NaiveBayesException(Exception):
	pass

class NaiveBayesCrossValidationException(Exception):
	pass


class CrossValCase(object):

	ALLOWED_REPRESENTATIONS = [ 
		'as_tficf',
		'as_modified_tficf',
		'as_frequencies',
		'as_tfidf'
	]
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
		limit=None
	):

		assert(representation in self.ALLOWED_REPRESENTATIONS)
		data_manager = dp.Data(limit=limit)

		# this requests the data manager to calculate the representation
		# specified by the parameters passed to the run method
		data = getattr(data_manager, representation)(
			use_cache=True,
			lemmatize=lemmatize, 
			find_specials=find_specials,
			remove_stops=remove_stops,
			use_digrams=use_digrams,
			data_part='train',
			as_vect=False
		)

		cross_val_tester = CrossValTester(data)
		accuracy = cross_val_tester.cross_validate(self.K)

		return accuracy


class CrossValTester(object):
	'''
	Given a data set, allows one to perform cross validation on the 
	NaiveBayesTextClassifier.

	The dataset must be a list of examples, where each example has the form

		(<id>, <feature_counts>, <class_name>)

	id should be an integer, feature_counts should be a dictionary with 
	numeric values, and class_name should be a string.

	So the dataset should look something like this:

		[
			...
			(1, {'feature1':3, 'feature2':1, ...}, 'physics'),
			...
		]
	'''

	def __init__(self, dataset, do_limit=False):

		# copy the data, initialize state
		self.dataset = copy.copy(dataset)
		# Randomize the examples' ordering.
		random.shuffle(self.dataset)

		# optionally limit data for testing
		if do_limit:
			self.dataset = self.dataset[:5000]

		self.size = len(self.dataset)

		self.scores = None
		self.are_results_available = False

		# as a way to speed computation, we the NBClassifier supports 
		# removing examples from the training set, so that when testing for
		# a new fold, the test set can be removed rather than retraining on
		# the whole training set.  We therefore begin by training on the whole 
		# set.  This has been tested to ensure it does not cause "pollution".
		self.classifier = NaiveBayesClassifier()
		self.classifier.train(self.dataset)


	def extract_test_set(self, fold, test_set_size, is_last=True):
		'''
		this was made as separate function to provide a testable interface.
		This removes a subset of examples from each class to be used as a 
		test set, while ensuring that the examples are not used for training.
		'''

		test_set = []

		# select the examples to be used as the test set for this class
		start = fold * test_set_size
		end = start + test_set_size
		if is_last:
			end = None

		test_set = self.dataset[start:end]

		# ensure that these examples are excluded from the training set
		for example in test_set:
			self.classifier.remove_example(example)

		return test_set


	def put_test_set_back_in_training_set(self, test_set):
		for example in test_set:
			self.classifier.add_example(example)


	def generate_crossval_predictions(self, k=None):

		# perform cross validation.  The predictions made during this routine
		# get stored in self.predictions
		self.cross_validate(k)

		# return predictions (sorted by id)
	 	self.predictions.sort()
		return self.predictions


	def cross_validate(self, k=None):
		'''
		divide the data set set into k equal folds (if k doesn't divide the
		number of examples in each class, then the folds won't all be equal).
		for each 'fold' of cross validation, train a NaiveBayesTextClassifier
		on all the data outside the fold, then test it on the data inside the
		fold, and repeat for all folds.  Keep a running tally of the number
		of classifications correct, for each class.
		'''

		# if k is not specified, do leave-one-out cross validation
		# i.e. have as many folds as examples
		if k is None:
			k = self.size

		k = int(k)

		# Make sure k is within the allowed range
		if k < 2:
			raise NaiveBayesCrossValidationException('You must have more than '
				'one fold for the cross validation test.')
		if k > self.size:
			raise NaiveBayesCrossValidationException('Their cannot be more '
				'folds than there are examples in each class!')

		test_set_size = self.size / k

		# to save computation time, the NaiveBayesClassifier supports 
		# "removing" examples, which limits the number of conditional 
		# probabilities that need to be re-calculated in each fold. To 
		# take advantage of this, we first train on the full dataset.
		# then selectively remove the examples on which we wish to test.
		# Testing shows that there is no pollution from doing this.
		self.score = 0
		self.predictions = []

		for fold in range(k):

			print 'fold %i' % fold

			is_last = bool(fold == k - 1)
			test_set = self.extract_test_set(fold, test_set_size, is_last)

			for example in test_set:

				example_id, example_features, example_class = example
				prediction = self.classifier.classify(example_features)
				self.predictions.append((example_id, prediction))

				if prediction == example_class:
					self.score += 1

			self.put_test_set_back_in_training_set(test_set)

		# return the overall accuracy.  
		# Other performance metrics are available through method calls.
		return self.score / float(self.size)



class NaiveBayesClassifier(object):
	'''
	Training consists of providing the classifier a set of examples having
	the structure:

		('class-name', 'word0', 'word1', 'word2', 'word3', 'word4')

	These can be added in bulk using train(), or singly using add_example().

	Examples can also be quickly removed: because they are stored as tuples
	of strings, they can be hashed, so this can be done quickly.
	'''

	IMPOSSIBLE = None

	def __init__(self):

		# global counts is used to keep track of the set of all features
		# if a given feature count goes down to zero, then we don't consider
		# it as existing anymore
		self.global_feature_counts = Counter()
		self.feature_counts = defaultdict(lambda: Counter())
		self.class_counts = Counter()
		self.num_examples = 0

		self.is_num_features_fresh = False
		self._num_features = 0

		self.is_num_classes_fresh = False
		self._num_classes = 0


	def train(self, examples):
		for example in examples:
			self.add_example(example)

	
	def add_example(self, example):
		class_name = example[2]
		features = example[1]
		
		self.num_examples += 1
		self.class_counts[class_name] += 1

		self.global_feature_counts.update(features)
		self.feature_counts[class_name].update(features)

		self.is_num_features_fresh = False
		self.is_num_classes_fresh = False


	def remove_example(self, example):
		class_name = example[2]
		features = example[1]

		self.num_examples -= 1
		self.class_counts[class_name] -= 1
		self.global_feature_counts.subtract(features)
		self.feature_counts[class_name].subtract(features)

		self.is_num_features_fresh = False
		self.is_num_classes_fresh = False


	def get_num_features(self):
		if self.is_num_features_fresh:
			return self._num_features

		num_features = 0
		for key, val in self.global_feature_counts.iteritems():
			if val>0:
				num_features += 1 

		self._num_features = num_features
		self.is_num_features_fresh = True

		return self._num_features


	def get_num_classes(self):
		if self.is_num_classes_fresh:
			return self._num_classes

		self.refresh_class_counts()
		return self._num_classes


	def refresh_class_counts(self):
		self._num_classes = 0
		self._class_names = set()
		for key, val in self.class_counts.items():
			if val>0:
				self._num_classes += 1
				self._class_names.add(key)

		self.is_num_classes_fresh = True


	def get_class_names(self):
		if self.is_num_classes_fresh:
			return self._class_names

		self.refresh_class_counts()
		return self._class_names


	def get_cond_prob(self, feature, class_name, use_add_one_smoothing=True):
		num_features = self.get_num_features()
		num_classes = self.get_num_classes()

		counts_for_feature_in_class = self.feature_counts[class_name][feature]
		num_examples_in_class = self.class_counts[class_name]

		# The technique "add-one-smoothing" helps calculate a "reasonable" 
		# likelihood for features that were never observed for a given class
		if use_add_one_smoothing:
			counts_for_feature_in_class += 1
			num_examples_in_class += num_features


		return counts_for_feature_in_class / float(num_examples_in_class)


	def get_prior(self, class_name):
		num_examples_in_class = self.class_counts[class_name]
		return num_examples_in_class / float(self.num_examples)


	def classify(self, example_features):
		'''
		Takes in an example_feature vector (which is missing the first 
		component, the class_name), and outputs the likelihood maximizing
		class, based on the assumption that features are independant of one
		another.
		'''

		# for each class, calculate a score equal to the likelihood that 
		# that class would produce this feature vector
		class_scores = defaultdict(lambda: 0)
		for class_name in self.get_class_names():
			for feature in example_features:

				feature_count = example_features[feature]
				cond_prob = self.get_cond_prob(feature, class_name)

				# summing log likelihoods is like multiplying.
				class_scores[class_name] += feature_count * np.log(cond_prob)

		# "multiply" each class's score by the class' prior probability 
		for class_name in class_scores:
			try:
				class_scores[class_name] += np.log(self.get_prior(class_name))
			except TypeError:
				print class_name

		# Which class was most likely? Remove impossible classes, then sort.
		feasible_class_scores = filter(
			lambda c: c[1] is not self.IMPOSSIBLE,
			class_scores.items()
		)
		feasible_class_scores.sort(None, lambda c: c[1], True)
		try:
			predicted_class, top_score = feasible_class_scores[0]
		except IndexError:
			predicted_class = self.class_counts.keys()[0]

		return predicted_class


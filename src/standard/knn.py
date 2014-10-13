from collections import Counter
import operator
import random

# Add this project's src folder to the path
import sys
import os
sys.path.append(os.path.abspath('../'))

import data_processing as dp


class kNNClassifer (object):
	'''
	k-nearest neighbor classifer that trains on examples of the form:

		('id', {'word1': 2, 'word2':1, 'word3':1, ...}, 'class_name')

	'''


	DISTS = [
		'EUCLID_DIST',
		'COSINE_DIST'
	]


	def __init__ (self, k, dist='EUCLID_DIST', normalize=False):
		self.k = k

		# Distance metric selection
		if dist == 'EUCLID_DIST':
			self.dist = 'calc_euclid_dist'
		elif dist== 'COSINE_DIST':
			self.dist = 'calc_cosine_dist'

		self.features = set([])
		self.examples = []


	def get_num_features (self):
		'''
		Get the number of features, i.e. words, in the dataset.
		'''

		return len(self.features)


	def calc_euclid_dist (self, example1, example2):
		'''
		Calculate the Euclidean distance between two example vectors.
		'''

		(word_dict1, word_dict2) = (example1[1], example2[1])

		dist_squared = 0
		for feature in self.features:
			x = word_dict1[feature] if feature in word_dict1 else 0
			y = word_dict2[feature] if feature in word_dict2 else 0
			dist_squared += (x - y) ** 2

		return dist_squared ** 0.5


	def calc_cosine_dist (self, example1, example2):
		'''
		Calculate the cosine distance between two example vectors, which
		is their cosine similiarity subtracted from 1.
		'''

		def calc_norm (word_dict):
			sum_sqrd = 0
			for feature in self.features:
				x = word_dict[feature] if feature in word_dict else 0
				sum_sqrd += x ** 2
			return sum_sqrd ** 0.5

		def calc_dot_prod (word_dict1, word_dict2):
			dot_prod = 0
			for feature in self.features:
				x = word_dict1[feature] if feature in word_dict1 else 0
				y = word_dict2[feature] if feature in word_dict2 else 0
				dot_prod += x * y
			return dot_prod

		(word_dict1, word_dict2) = (example1[1], example2[1])
		dot_prod = calc_dot_prod(word_dict1, word_dict2)
		norm1 = calc_norm(word_dict1)
		norm2 = calc_norm(word_dict2)

		return 1 - float(dot_prod) / (norm1 * norm2)


	def normalize (self):
		'''
		Normalize the feature values in the dataset by feature.
		'''

		for feature in self.features:
			feature_vals = ( example[1][feature] for example in self.examples )
			
			# Max. and min. values for the given feature
			(mx, mn) = (max(feature_vals), min(feature_vals))
			
			multiplier = 1.0 / (mx - mn)

			self.examples = copy.deepcopy(self.examples)
			for example in self.examples:
				# Subtract min. from each feature value and apply multiplier
				example[1][feature] = multiplier * (example[1][feature] - mn)


	def add_example (self, example):
		'''
		Add single example to dataset.
		'''

		word_dict = example[1]
	
		for w in word_dict:
			self.features.add(w)

		self.examples.append(example)


	def train (self, examples):
		'''
		Train on set of examples by adding them to dataset.
		'''

		for example in examples:
			self.add_example(example)


	def get_nearest_neighbours (self, example):
		'''
		Get the k-nearest neighbours in the dataset to the arguemnt example.
		'''

		# Get the nearest neighbour that is farthest from the argument
		# example in the current list of nearest neighbours.
		def get_max_nearest_neighbour ():
			max_nearest_neighbour = None
			for n_id, dist_and_class in nearest_neighbours.iteritems():
				d = dist_and_class[0]
				if max_nearest_neighbour is None:
					max_nearest_neighbour = (n_id, d)
				else:
					if max_nearest_neighbour[1] < d:
						max_nearest_neighbour = (n_id, d)
			return max_nearest_neighbour

		calc_dist = getattr(self, self.dist)

		# Nearest neighbours indexed by example id with distances from
		# the example as values
		nearest_neighbours = {}

		for e in self.examples:
			dist = calc_dist(e, example)
			
			e_id = e[0]
			e_class_name = e[2]
			if len(nearest_neighbours) < self.k:
				nearest_neighbours[e_id] = (dist, e_class_name)
			else:
				(mnn_id, mnn_dist) = get_max_nearest_neighbour()
				if dist < mnn_dist:
					nearest_neighbours[e_id] = (dist, e_class_name)
					del nearest_neighbours[mnn_id]

		return nearest_neighbours


	def classify(self, example):
		'''
		Takes in an example (which is missing the class), and outputs the
		majority class of the k nearest neighbors.
		'''

		nearest_neighbours = self.get_nearest_neighbours(example)

		# Set of classes that appear in the nearest neighbours.
		class_names = [ dist_and_class[1] for nn_id, dist_and_class in nearest_neighbours.iteritems() ]
		class_counter = Counter(class_names)

		# Frequency of the majority class(es).
		num_majority = class_counter.most_common(1)[0][1]
		# Classes tied for majority.
		majority_classes = [ c for c in class_counter if class_counter[c] is num_majority ]

		# Sum of the nearest neighbour distances by class for the classes
		# tied for majority.
		majority_class_dists = dict.fromkeys(majority_classes, 0.0)
		for nn_id, dist_and_class in nearest_neighbours.iteritems():
			(dist, class_name) = dist_and_class
			if class_name in majority_classes:
				majority_class_dists[class_name] += dist

		# Return the class with the lowest nearest neighbour distance sum
		# as the prediction.
		predicted_class = min(majority_class_dists.iteritems(), key=operator.itemgetter(1))[0]
		return predicted_class


class CrossValTester(object):
	'''
	Given a data set, allows one to perform cross validation with the 
	kNN classifier.

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

	# The min. and max. k values to consider for the k-nearest neighbours
	# algorithm
	MIN_NN_K = 5
	MAX_NN_K = 30


	def __init__ (self, dist, dataset, limit=None):
		self.dist = dist

		self.dataset = dataset

		# Randomize the examples' ordering.
		random.shuffle(self.dataset)

		# Optionally limit data for improved time efficiency
		if limit is not None:
			self.dataset = self.dataset[:limit]

		self.size = len(self.dataset)
		self.scores = None


	def extract_train_and_test_set (self, fold, test_set_size, is_last=True):
		'''
		Partitions the examples into a training and test set.
		'''

		# Select the index range of examples to be used as the test set
		startIdx = fold * test_set_size
		endIdx = startIdx + test_set_size
		if is_last:
			endIdx = None

		test_set = self.dataset[startIdx:endIdx]

		# Select the remaining examples for the training set
		train_set = self.dataset[:startIdx] + self.dataset[endIdx:]

		return (train_set, test_set)


	def extract_subtrain_and_heldout_set (self, train_set, heldout_set_size):
		'''
		Partitions the training set into a sub-training and held-out set.
		'''

		heldout_set = train_set[:heldout_set_size]
		subtrain_set = train_set[heldout_set_size:]
		return (subtrain_set, heldout_set)


	def train_and_predict (self, nn_k, train_set, test_set):
		'''
		Train the classifier on the training set and predict on the test
		set.
		'''

		classifier = kNNClassifer(k=nn_k, dist=self.dist)
		classifier.train(train_set)

		score = 0
		for example in test_set:
			prediction_class = classifier.classify(example)
			if prediction_class == example[2]:
				score += 1

		# Return the accuracy
		return float(score) / len(test_set)


	def det_opt_k (self, subtrain_set, heldout_set):
		'''
		Determine the optimal value of k for the k-nearest neighbours
		algorithm by iterating over a range of values for k and training
		on the sub-training set and testing on the held-out set.
		'''

		opt_k = None
		max_accuracy = 0.0

		# Range k from MIN_NN_K to MAX_NN_K
		for nn_k in range(
			CrossValTester.MIN_NN_K,
			CrossValTester.MAX_NN_K + 1
		):
			score = 0

			accuracy = self.train_and_predict(
				nn_k = nn_k,
				train_set=subtrain_set,
				test_set=heldout_set
			)

			# Set the optimal value of k to the value producing the highest
			# accuracy.
			if max_accuracy < accuracy:
				opt_k = nn_k
				max_accuracy = accuracy

		return opt_k


	def cross_validate(self, k=None):
		'''
		Divide the dataset set into k equal folds (If k doesn't divide the
		number of examples evenly, then the folds won't all be equal). For
		each 'fold' of cross validation, train a kNN classifier on all the
		data outside the fold, then test it on the data inside the
		fold, and repeat for all folds.  Keep a running tally of the number
		of classifications correct. Training proceeds by first determining
		an optimal value of nn_k for the kNN classifier by dividing the
		training set into a sub-training and held-out set and running
		through a range of different values for nn_k. Once the optimal
		value is determined, generate a classifier for with the optimal
		value and train it on the entire training set before predicting
		on the test set, i.e. 'fold'.
		'''


		# If k is not specified, do leave-one-out cross validation
		if k is None:
			k = self.size
		k = int(k)

		test_set_size = self.size / k

		self.score = 0

		for fold in range(k):

			print 'Fold %i' % fold

			is_last = (fold is k - 1)
			(train_set, test_set) = self.extract_train_and_test_set(fold, test_set_size, is_last)

			# Size of the held-out set is 10% of the total number of
			# training examples or 100 examples, whichever is smaller.
			tenth_of_examples = int(0.1 * len(train_set))
			heldout_set_size = tenth_of_examples if tenth_of_examples < 100 else 100

			# Get sub-training and held-out sets.
			(subtrain_set, heldout_set) = self.extract_subtrain_and_heldout_set(
				train_set=train_set,
				heldout_set_size=heldout_set_size
			)

			# Determine the optimal value for nn_k.
			opt_k = self.det_opt_k(
				subtrain_set=subtrain_set,
				heldout_set=heldout_set
			)

			print '\tOpt. NN k: %i' % opt_k

			# Create and traing kNN classifier.
			classifier = kNNClassifer(k=opt_k, dist=self.dist)
			classifier.train(train_set)

			print '\t# Features: %i' % classifier.get_num_features()

			# Apply classifier on the test set and tally score
			for example in test_set:
				prediction_class = classifier.classify(example)
				if prediction_class == example[2]:
					self.score += 1

		accuracy = self.score / float(self.size)

		print 'OVERALL ACCURACY: %f' % accuracy

		# Return the overall accuracy 
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
		dist=,
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

		cross_val_tester = CrossValTester(dist=dist, dataset=dataset)
		accuracy = cross_val_tester.cross_validate(self.K)

		return accuracy


if __name__ == '__main__':
	print CrossValCase().run(
		dist='COSINE_DIST',
		representation='as_trimmed_tficf',
		lemmatize=True,
		find_specials=True,
		remove_stops=True,
		use_digrams=False,
		limit=500
	)

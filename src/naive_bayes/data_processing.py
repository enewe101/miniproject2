from itertools import izip
import numpy as np
import csv
import os
import json
import re
import math

try:
	import nltk
except ImportError:
	pass

from collections import defaultdict, Counter


def merge_input_output():
	input_reader = csv.reader(open('data/raw/train_input.csv', 'r'))
	output_reader = csv.reader(open('data/raw/train_output.csv', 'r'))
	writer = csv.writer(open('data/raw/train.csv', 'w'))

	for input_row, output_row in izip(input_reader, output_reader):
		writer.writerow(input_row + output_row[1:])



class Data(object):

	'''
		to use this file, make sure that the following raw data files
		have been placed in /data/raw:

			- train_input.csv
			- train_output.csv
			- test_input.csv

	'''

	# some useful regular expressions
	NON_ALPHA = re.compile(r'[^a-zA-Z]+')
	NON_ALPHANUM = re.compile(r'[^a-zA-Z0-9_]+')
	NUMBER = re.compile(r'\d+')

	# Where should things be stored and read?
	FNAMES = {
		'raw':'data/raw/train.csv',
		'as_frequencies': 'data/cache/frequencies.json',
		'lemmatized': 'data/cache/lemmatized.json',
		'hypernyms': 'data/cache/hypernyms.json',
		'as_tficf': 'data/cache/as_tficf.json',
		'class_counts': 'data/cache/class_counts.json',
		'cf_scores': 'data/cache/cf_scores.json',
		'icf_scores': 'data/cache/icf_scores.json',
		'tf_idf' : 'data/cache/tf_idf.json'
	}

	# Keep things in cache so we don't need to keep calculating
	CACHE = {}

	# Some flags
	NOT_CACHED = 0
	FILE_CACHED = 1
	MEM_CACHED = 2

	# Other constants
	NUM_CLASSES = 4

	def __init__(self, limit=None, fname="data/raw/train.csv"):

		# check if the merged data file exists, if not, make it
		if not os.path.isfile(self.FNAMES['raw']):
			print ('merging training input and output on the first '
				'use of Data...')
			merge_input_output()

		# load raw data into memory
		reader = csv.reader(open(fname, 'r'))

		if limit is not None:
			self.data = []
			for i in range(limit):
				row = reader.next()
				self.data.append(tuple(row))

		else:
			self.data = [ tuple(row) for row in reader ] 


	def check_cache(self, name):

		# did we calculate this already during this session, is it in memory?
		if name in self.CACHE:
			return self.MEM_CACHED

		# did we calculate this in a previous session, was it written to file?
		if os.path.isfile(self.FNAMES[name]):
			return self.FILE_CACHED

		# no, it's not cached
		return self.NOT_CACHED


	def get_cache(self, name):

		if name in self.CACHE:
			print 'using %s found in memory cache' % name
			return self.CACHE[name]

		if os.path.isfile(self.FNAMES[name]):
			print 'loading %s from file cache' % name
			return json.loads(open(self.FNAMES[name]).read())

		raise IOError('%s was not found in the cache' % name)
		 

	def cache(self, name, data):

		fh = open(self.FNAMES[name], 'w')
		fh.write(json.dumps(data, indent=2))
		self.CACHE[name] = data
		fh.close()


	def lemmatized(self, use_cache=True):
		'''
		uses a frequency-representation (see `self.as_frequencies`).
		words get stemmed before counting.
		'''
		# check for cached data first
		if use_cache and self.check_cache('lemmatized'):
			return self.get_cache('lemmatized')

		lmtzr = nltk.stem.wordnet.WordNetLemmatizer()

		return_data = []
		for row in self.data:

			# split on any series of characters that aren't letters, 
			# numbers or the underscore
			features = self.NON_ALPHANUM.split(row[1])
			features = Counter(map(lambda x: lmtzr.lemmatize(x), features))

			# get rid of empty-string features
			del features['']

			class_name = row[2]
			return_data.append((class_name, dict(features)))

		self.cache('lemmatized', return_data)

		return return_data


	def hypernyms(self, use_cache=True):
		'''
		uses a frequency-representation (see `self.as_frequencies`).
		we include the hypernyms of words (using wordnet)
		'''
		# check for cached data first
		if use_cache and self.check_cache('hypernyms'):
			return self.get_cache('hypernyms')

		print 'calculating hypernym counts...'

		w = nltk.corpus.wordnet

		return_data = []
		for row in self.data:

			# split on any series of characters that aren't letters, 
			# numbers or the underscore
			features = self.NON_ALPHANUM.split(row[1])

			# get the synsets, then get their hypernyms
			synsets = reduce(lambda x,y: x + w.synsets(y), features, [])
			hypernyms = reduce(lambda x,y: x + [y] + y.hypernyms(), 
				synsets, [])
			wordnet_features = reduce(
				lambda x,y: x + [l.name for l in y.lemmas], hypernyms, []
			)

			features = features + wordnet_features

			# add the synsets and hypernyms (by name) to the as-found words
			features = Counter(features)

			# get rid of empty-string features
			del features['']

			class_name = row[2]
			return_data.append((class_name, dict(features)))

		self.cache('hypernyms', return_data)

		return return_data


	def get_class_counts(self, use_cache=True):

		# check for cached data first
		if use_cache and self.check_cache('class_counts'):
			return self.get_cache('class_counts')

		print 'calculating class-counts...'

		class_counts = defaultdict(lambda: Counter())
		for row in self.data:

			class_name = row[2]

			# update word counts for the class from which this example came
			class_counts[class_name].update(self.NON_ALPHANUM.split(row[1]))

		# delete the empty-string counts
		for class_name in class_counts:
			del class_counts[class_name]['']

		self.cache('class_counts', class_counts)

		return class_counts


	def compute_cf_scores(self, use_cache=True):
		'''
			This computes the class-frequency for words, that is, for each
			word in the vocabulary of the corpus, how many classes does it
			appear in.  Since we have 4 classes, this has to be a number from
			1 to 4.
		'''

		# check for cached data first
		if use_cache and self.check_cache('cf_scores'):
			return self.get_cache('cf_scores')

		print 'calculating cf_scores...'

		# first, we must find the counts on a per-class basis
		class_counts = self.get_class_counts(use_cache)

		# now we are ready to compute icf weighting factors
		# first, assemble the entire vocabulary
		vocab = set()
		for class_name in class_counts:
			vocab.update(class_counts[class_name].keys())

		print 'total size of vocabulary: %d words.' % len(vocab)

		icf_scores = {}
		for word in vocab:

			# how many categories have this word?
			presence = [
				bool(word in class_counts[cn] and class_counts[cn][word]>0) 
				for cn in class_counts
			]
			icf_scores[word] = sum(presence)

		self.cache('cf_scores', icf_scores)

		return icf_scores


	def compute_icf_scores(self, use_cache=True):
		'''
			This computes the inverse class frequency score for all the words
			in the vocabulary of the corpus.  This is a kind of relevancy
			weighting.
		'''

		# check for cached data first
		if use_cache and self.check_cache('icf_scores'):
			return self.get_cache('icf_scores')

		print 'calculating icf_scores...'

		# get cf_scores and convert to icf_scores
		cf_scores = self.compute_cf_scores(use_cache)
		icf_scores = {}
		for word in cf_scores:
			icf_scores[word] = np.log2(
				1 + self.NUM_CLASSES / float(cf_scores[word]))

		# cache the icf_scores
		self.cache('icf_scores', icf_scores)

		return icf_scores


	def as_tficf(self, data_part='train', use_cache=True):

		# check for cached data first
		if use_cache and self.check_cache('as_tficf'):
			return self.get_cache('as_tficf')

		# compute the icf_scores
		icf_scores = self.compute_icf_scores(use_cache)

		# now get the raw word frequencies
		as_frequencies_data = self.as_frequencies(use_cache)

		print 'calculating tficf\'s...'

		# for each item in the dataset, multiply word frequency by icf_score
		return_data = []
		for item in as_frequencies_data:

			# unpack the item
			idx, frequencies, class_name = item

			# multiply word frequencies by their corresponding icf_scores
			tficf_scores = {}
			for word in frequencies:
				tficf_scores[word] = frequencies[word] * icf_scores[word]

			return_data.append([idx, tficf_scores, class_name])

		# cache the results
		self.cache('as_tficf', return_data)

		return return_data


	def as_frequencies(self, use_cache=True):
		'''
		represent the data using the counts structure
			[
				('class_name', {'word1': 2, 'word2':1, 'word3':1, ...}),
				...
			]
		'''

		# check for cached data first
		if use_cache and self.check_cache('as_frequencies'):
			return self.get_cache('as_frequencies')

		print 'calculating frequencies...'

		return_data = []
		for row in self.data:

			# split on any series of characters that aren't letters, 
			# numbers or the underscore
			features = Counter(self.NON_ALPHANUM.split(row[1]))

			# get rid of empty-string features
			del features['']

			idx = row[0]
			class_name = row[2]
			return_data.append((idx, dict(features), class_name))

		self.cache('as_frequencies', return_data)

		return return_data

	def as_tfidf(self, use_cache=True):

		# check for cached data first
		if use_cache and self.check_cache('tf_idf'):
			return self.get_cache('tf_idf')

		# frequency of each word in each document
		word_counts = self.as_frequencies()

		# number of words
		n_documents = len(word_counts)

		# number of documents containing a word
		global_counts = Counter()

		return_data = []

		# populating global_counts
		for element in word_counts:
			idx, feature_counts, class_name = element
			global_counts.update(feature_counts.keys())

		# generating tf-idf
		for element in word_counts:
			idx, feature_counts, class_name = element
			n_words = sum(feature_counts.values())

			new_counts = {}
			for item in feature_counts:
				tf = float(feature_counts[item])/float(n_words)
				df = float(global_counts[item])/float(n_documents)
				idf = - math.log(df)
				new_counts[item] = tf*idf
			return_data.append((idx,new_counts,class_name))

		self.cache('tf_idf',return_data)
		return return_data







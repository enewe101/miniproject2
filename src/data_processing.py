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


# Makes the merged data file for use by the Data class.  
# This normally does not need to be called directly...
def merge_input_output():
	input_reader = csv.reader(open('../data/raw/train_input.csv', 'r'))
	output_reader = csv.reader(open('../data/raw/train_output.csv', 'r'))
	writer = csv.writer(open('../data/raw/train_merged.csv', 'w'))

	# write a new file that merges the info from both
	for input_row, output_row in izip(input_reader, output_reader):
		writer.writerow(input_row + output_row[1:])


# Decorator that helps in building the Data class
def add_caching(f):
	f_name = f.__name__

	def augmented_with_caching(
		self, 
		use_cache=True,
		lemmatize=False,
		find_specials=False,
		remove_stops=False,
		use_digrams=False,
		data_part='train'
	):

		cache_address =  (
			data_part, f_name, lemmatize, find_specials, remove_stops, 
			use_digrams, self.limit)

		# check if results for this calculation were cached
		if use_cache and self.check_cache(cache_address):
			return self.get_cache(cache_address)

		# if not cached, calculate the result using the underlying function
		return_data = f(
			self, use_cache, lemmatize, find_specials, remove_stops, 
			use_digrams, data_part)

		# add the result of the calculation to the cache
		self.cache(cache_address, return_data)

		return return_data

	return augmented_with_caching


# Decorator that helps in building the Data class
def enable_vectors(f):
	def augmented_with_vectors(
		self, 
		use_cache=True,
		lemmatize=False,
		find_specials=False,
		remove_stops=False,
		use_digrams=False,
		data_part='train',
		as_vect=False,
	):
		result_data= f(
			self, use_cache, lemmatize, find_specials, remove_stops, 
			use_digrams, data_part)

		# if as_vect is not requested, just return the results normally
		if not as_vect:
			return result_data

		## otherwise, return a VectorList iterator...

		# first get the alphabetically ordered vocabulary
		vocab = self.get_vocab(
			use_cache, lemmatize, find_specials, remove_stops, use_digrams)

		# now make the VectorList iterator, and return it
		return VectorList(result_data, vocab)

	return augmented_with_vectors


# This class provides a data management API
class Data(object):

	'''
		to use this class, make sure that the following raw data files
		have been placed in /data/raw:

			- train_input.csv
			- train_output.csv
			- test_input.csv

		This class provides a data loading and transforming service.
		It reads a raw form of the abstracts dataset, and returns a 
		more convenient structured form.  The format of the data is always
		the same, but there are various pre-processing and weighting 
		options.  The data format is a list of 'examples', and each example
		is a tuple, having the format (`id`, `frequencies`, `class_name`).
		So it looks something like this:

			[
				...
				('class_name', {'word1': 2, 'word2':1, 'word3':1, ...}),
				...
			]

		You can get output like that by calling any of the methods that
		match 'as_*'.  For example, `as_frequencies()` returns the data
		with the frequencies as, well, the actual frequency of occurence of
		the word in the given abstract, and 'as_tficf()' returns the 
		frequencies multiplied by a weighting factor.

		All methods accept some pre-processing options, including: 
		'normalize', 'lemmatize', and 'remove_stops'.  These are all False
		by default.

		All the methods also accept a `data_part` keyword argument, which 
		can take the values 'test' or 'train' (it is 'test' by default).
		This determines whether the testing or training data will be returned.
	'''

	# Some useful regular expressions
	NON_ALPHA = re.compile(r'[^a-zA-Z]+')
	NON_ALPHANUM = re.compile(r'[^a-zA-Z0-9_]+')
	NUMBER = re.compile(r'\b\d+(\.\d+)?\b')
	LATEX = re.compile(r'\\[a-zA-Z]+{[^}]+}')
	URL = re.compile(
		r'\b(http://)?([a-zA-Z0-9_]+\.)+(com|edu|net|org|ca|us)\b')

	# Keep things in cache so we don't need to keep calculating
	CACHE = {}

	# Some flags
	NOT_CACHED = 0
	FILE_CACHED = 1
	MEM_CACHED = 2

	# Other constants
	NUM_CLASSES = 4
	CACHE_DIR = 'cache'
	RAW_DIR = 'raw'
	RAW_MERGED = os.path.join(RAW_DIR, 'train_merged.csv')
	RAW_INPUT = os.path.join(RAW_DIR, 'train_input.csv')
	RAW_OUTPUT = os.path.join(RAW_DIR, 'train_output.csv')
	RAW_TEST = os.path.join(RAW_DIR, 'test_input.csv')


	def __init__(self, 
		data_dir='../../data',
		limit=None,
		verbose=True,
		use_mem_cache=False
	):

		self.data_dir = data_dir
		self.verbose = verbose
		self.limit = limit
		self.use_mem_cache

		# check if the merged data file exists, if not, make it
		if not os.path.isfile(os.path.join(data_dir, self.RAW_MERGED)):
			self.say('merging training input and output on the first '
				'use of Data...')
			merge_input_output()

		# load raw data into memory
		reader_train = csv.reader(
			open(os.path.join(data_dir, self.RAW_MERGED), 'r'))
		reader_test = csv.reader(
			open(os.path.join(data_dir, self.RAW_TEST), 'r'))

		# the files have headers, so advance both readers by one line
		reader_train.next()
		reader_test.next()

		# limit can be used to limit the amount of data used
		if limit is not None:
			self.data = [tuple(reader_train.next()) for i in range(limit)]
			self.test_data = [tuple(reader_test.next()) for i in range(limit)]

		else:
			self.data = [tuple(row) for row in reader_train] 
			self.test_data = [tuple(row) for row in reader_train] 


	@enable_vectors
	@add_caching
	def as_tficf(
			self,
			use_cache=True,
			lemmatize=False,
			find_specials=False,
			remove_stops=False,
			use_digrams=False,
			data_part='train'
		):

		# compute the icf_scores
		icf_scores = self.compute_icf_scores(
			use_cache, lemmatize, find_specials, remove_stops, use_digrams)

		# now get the raw word frequencies
		as_frequencies_data = self.as_frequencies(
			use_cache, lemmatize, find_specials, remove_stops, use_digrams, 
			data_part)

		self.say('calculating tficf\'s...')

		# for each item in the dataset, multiply word frequency by icf_score
		return_data = []
		for item in as_frequencies_data:

			# unpack the item
			if data_part == 'train':
				idx, frequencies, class_name = item
			else:
				idx, frequencies = item


			# multiply word frequencies by their corresponding icf_scores
			tficf_scores = {}
			for word in frequencies:
				score = icf_scores[word] if word in icf_scores else 1.0
				tficf_scores[word] = frequencies[word] * score

			if data_part == 'train':
				return_data.append([idx, tficf_scores, class_name])
			else:
				return_data.append([idx, tficf_scores])


		return return_data


	@enable_vectors
	@add_caching
	def as_modified_tficf(
			self,
			use_cache=True,
			lemmatize=False,
			find_specials=False,
			remove_stops=False,
			use_digrams=False,
			data_part='train'
		):
		
		# compute the modified_icf_scores
		modified_icf_scores = self.compute_modified_icf_scores(
			use_cache, lemmatize, find_specials, remove_stops, use_digrams)

		# now get the raw word frequencies
		as_frequencies_data = self.as_frequencies(
			use_cache, lemmatize, find_specials, remove_stops, use_digrams,
			data_part)

		self.say('calculating modified_tficf\'s...')

		# for each item in the dataset, multiply word frequency by icf_score
		return_data = []
		for item in as_frequencies_data:

			# unpack the item
			if data_part == 'train':
				idx, frequencies, class_name = item
			else:
				idx, frequencies = item

			# multiply word frequencies by their corresponding 
			# modified_icf_scores
			modified_tficf_scores = {}
			for word in frequencies:
				modified_tficf_scores[word] = (
					frequencies[word] * modified_icf_scores[word])

			if data_part == 'train':
				return_data.append([idx, modified_tficf_scores, class_name])
			else:
				return_data.append([idx, modified_tficf_scores])

		return return_data


	@enable_vectors
	@add_caching
	def as_frequencies(
		self, 
		use_cache=True,
		lemmatize=False,
		find_specials=False,
		remove_stops=False,
		use_digrams=False,
		data_part='train'
	):

		self.say('calculating frequencies...')

		# get some helper objects, if we plan to lemmatize or remove stop words
		if lemmatize:
			lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
		if remove_stops:
			stops = set(nltk.corpus.stopwords.words('english'))

		# use the test set or training set, depending on what was requested
		if data_part == 'train':
			data_set = self.data
		elif data_part == 'test':
			data_set = self.test_data
		else:
			raise ValueError("data_part must be either 'train' or 'test'")

		return_data = []
		for row in data_set:

			if data_part == 'train':
				idx, features, class_name = row
			else:
				idx, features = row

			# make everything lowercase
			features = features.lower()

			# maybe find special words and replace them with a special token
			if find_specials:
				features = self.NUMBER.sub('aNUMBER', features)
				features = self.URL.sub('aURL', features)
				features = self.LATEX.sub('aLATEX', features)

			# split on any characters that aren't alphanumeric, or underscore
			features = self.NON_ALPHANUM.split(features)

			# maybe remove stopwords
			if remove_stops:
				features = [f for f in features if f not in stops]

			# maybe lematize
			if lemmatize:
				features = [lmtzr.lemmatize(f) for f in features]

			# maybe get digrams
			if use_digrams:
				digrams = ['%s %s' % (features[i], features[i+1]) 
					for i in range(len(features)-1)]
				features += digrams

			# convert into counts
			features = Counter(features)

			# get rid of empty-string features
			del features['']

			if data_part == 'train':
				return_data.append((idx, dict(features), class_name))
			else:
				return_data.append((idx, dict(features)))

		return return_data


	@enable_vectors
	@add_caching
	def as_tfidf(
			self,
			use_cache=True,
			lemmatize=False,
			find_specials=False,
			remove_stops=False,
			use_digrams=False,
			data_part='train'
		):

		# frequency of each word in each document
		word_counts = self.as_frequencies(
			use_cache, lemmatize, find_specials, remove_stops, use_digrams,
			data_part)

		# number of words
		n_documents = len(word_counts)

		# number of documents containing a word
		global_counts = Counter()

		return_data = []

		# populating global_counts
		for element in word_counts:
			feature_counts = element[1]
			global_counts.update(feature_counts.keys())

		# generating tf-idf
		for element in word_counts:

			if data_part == 'train':
				idx, feature_counts, class_name = element
			else:
				idx, feature_counts = element

			n_words = sum(feature_counts.values())

			new_counts = {}
			for item in feature_counts:
				tf = float(feature_counts[item])/float(n_words)
				df = float(global_counts[item])/float(n_documents)
				idf = - math.log(df)
				new_counts[item] = tf*idf

			if data_part == 'train':
				return_data.append((idx,new_counts,class_name))
			else:
				return_data.append((idx,new_counts))

		return return_data


	@add_caching
	def get_vocab(
			self, 
			use_cache=True,
			lemmatize=False,
			find_specials=False,
			remove_stops=False,
			use_digrams=False,
			data_part='train'	# this has no effect, but is needed 
								# for compatibility with the caching aspect
		):

		self.say('calculating vocabulary...')

		# first get the class counts: we can get the vocabulary from this
		class_counts = self.get_class_counts(
			use_cache, lemmatize, find_specials, remove_stops, use_digrams)

		# now assemble the vocabulary from the class_counts
		vocab = set()
		for class_name in class_counts:
			vocab.update(class_counts[class_name].keys())

		vocab = list(vocab)
		vocab.sort()

		return vocab


	@add_caching
	def get_class_counts(
			self,
			use_cache=True,
			lemmatize=False,
			find_specials=False,
			remove_stops=False,
			use_digrams=False,
			data_part='train'	# this has no effect, but is needed 
								# for compatibility with the caching aspect
		):

		self.say('calculating class-counts...')

		class_counts = defaultdict(lambda: Counter())
		as_frequencies_data = self.as_frequencies(
			use_cache, lemmatize, find_specials, remove_stops, use_digrams)

		for row in as_frequencies_data:
			idx, frequencies, class_name = row

			# update word counts for the class from which this example came
			class_counts[class_name].update(frequencies)

		# delete the empty-string counts
		for class_name in class_counts:
			del class_counts[class_name]['']

		return class_counts


	@add_caching
	def compute_cf_scores(
			self, 
			use_cache=True,
			lemmatize=False,
			find_specials=False,
			remove_stops=False,
			use_digrams=False,
			data_part='train'	# this has no effect, but is needed 
								# for compatibility with the caching aspect
		):
		'''
			This computes the class-frequency for words, that is, for each
			word in the vocabulary of the corpus, how many classes does it
			appear in.  Since we have 4 classes, this has to be a number from
			1 to 4.
		'''

		self.say('calculating cf_scores...')

		# first, we must find the counts on a per-class basis
		class_counts = self.get_class_counts(
			use_cache, lemmatize, find_specials, remove_stops, use_digrams)

		# now we are ready to compute icf weighting factors
		# first, assemble the entire vocabulary
		vocab = set()
		for class_name in class_counts:
			vocab.update(class_counts[class_name].keys())

		self.say('total size of vocabulary: %d words.' % len(vocab))

		icf_scores = {}
		for word in vocab:

			# how many categories have this word?
			presence = [
				bool(word in class_counts[cn] and class_counts[cn][word]>0) 
				for cn in class_counts
			]
			icf_scores[word] = sum(presence)

		return icf_scores


	@add_caching
	def compute_icf_scores(
			self, 
			use_cache=True,
			lemmatize=False,
			find_specials=False,
			remove_stops=False,
			use_digrams=False,
			data_part='train'	# this has no effect, but is needed 
								# for compatibility with the caching aspect
		):
		'''
			This computes the inverse class frequency score for all the words
			in the vocabulary of the corpus.  This is a kind of relevancy
			weighting.
		'''

		self.say('calculating icf_scores...')

		# get cf_scores and convert to icf_scores
		cf_scores = self.compute_cf_scores(
			use_cache, lemmatize, find_specials, remove_stops)
		icf_scores = {}
		for word in cf_scores:
			icf_scores[word] = np.log2(
				1 + self.NUM_CLASSES / float(cf_scores[word]))

		return icf_scores


	@add_caching
	def compute_modified_icf_scores(
			self, 
			use_cache=True,
			lemmatize=False,
			find_specials=False,
			remove_stops=False,
			use_digrams=False,
			data_part='train'	# this has no effect, but is needed 
								# for compatibility with the caching aspect
		):
		'''
			This computes a modified version of the inverse class frequency 
			score for all the words in the vocabulary of the corpus.  This is 
			a kind of relevancy weighting.
		'''

		self.say('calculating modified_icf_scores...')

		# get cf_scores and convert to modified_icf_scores
		class_counts = self.get_class_counts(
			use_cache, lemmatize, find_specials, remove_stops, use_digrams)

		# assemble the entire vocabulary
		vocab = set()
		for class_name in class_counts:
			vocab.update(class_counts[class_name].keys())

		modified_icf_scores = {}
		for word in vocab:
			word_counts = [
				class_counts[cn][word] if word in class_counts[cn] else 0
				for cn in class_counts
			]
			modified_icf_scores[word] = np.log2(
				(max(word_counts) + 1) / float(min(word_counts) + 1)
			)

		return modified_icf_scores


	def say(self, string):
		if self.verbose:
			print string

	def as_vect(
		self,
		data,
		use_cache=True,
		lemmatize=False,
		find_specials=False, 
		remove_stops=False
	):
		'''
			provides an returns an iterator that vields vectors
		'''
		self.vocab = self.get_vocab(
			self, lemmatize, find_specials, remove_stops)

		word_list = list(self.vocab)
		word_list.sort()

		return VectorList(data, word_list)


	def check_cache(self, cache_address):

		# did we calculate this already during this session, is it in memory?
		if cache_address in self.CACHE:
			return self.MEM_CACHED

		# did we calculate this in a previous session, was it written to file?
		if os.path.isfile(self.get_cache_fname(cache_address)):
			return self.FILE_CACHED

		# no, it's not cached
		return self.NOT_CACHED


	def get_cache(self, cache_address):

		if cache_address in self.CACHE:
			self.say( 'using %s found in memory cache' 
				% self.get_cache_fname(cache_address))
			return self.CACHE[cache_address]

		if os.path.isfile(self.get_cache_fname(cache_address)):
			self.say('loading %s from file cache' 
				% self.get_cache_fname(cache_address))

			# read the data from file
			return_data = json.loads(
				open(self.get_cache_fname(cache_address)).read())

			# keep a reference in the memory cache
			self.CACHE[cache_address] = return_data

			return return_data

		raise IOError(
			'%s was not found in the cache' 
			% self.get_cache_fname(cache_address)
		)
		 

	def get_cache_fname(self, cache_address):

		data_part, func_name, lemmatize, find_specials, remove_stops, use_digrams, limit = cache_address
		f_name = data_part
		f_name += '_' + func_name
		f_name += '_lem' if lemmatize else ''
		f_name += '_spec' if find_specials else ''
		f_name += '_nostop' if remove_stops else ''
		f_name += '_digrams' if use_digrams else ''
		f_name += ('_%s' % str(self.limit)) if self.limit is not None else ''

		return os.path.join(self.data_dir, self.CACHE_DIR, '%s.json' % f_name)


	def cache(self, cache_address, data):
		write_fname = self.get_cache_fname(cache_address)
		fh = open(write_fname, 'w')
		fh.write(json.dumps(data, indent=2))
		if self.use_mem_cache:
			self.CACHE[cache_address] = data
		fh.close()




#	def lemmatized(self, use_cache=True):
#		'''
#			uses a frequency-representation (see `self.as_frequencies`).
#			words get stemmed before counting.
#		'''
#		# check for cached data first
#		if use_cache and self.check_cache('lemmatized'):
#			return self.get_cache(
#				('lemmatized', lemmatize, find_specials, remove_stops))
#
#		lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
#
#		return_data = []
#		for row in self.data:
#
#			# split on any series of characters that aren't letters, 
#			# numbers or the underscore
#			features = self.NON_ALPHANUM.split(row[1])
#			features = Counter(map(lambda x: lmtzr.lemmatize(x), features))
#
#			# get rid of empty-string features
#			del features['']
#
#			class_name = row[2]
#			return_data.append((class_name, dict(features)))
#
#		self.cache(
#			('lemmatized', lemmatize, find_specials, remove_stops), 
#			return_data
#		)
#
#		return return_data


#	def hypernyms(self, use_cache=True):
#		'''
#		uses a frequency-representation (see `self.as_frequencies`).
#		we include the hypernyms of words (using wordnet)
#		'''
#		# check for cached data first
#		if use_cache and self.check_cache('hypernyms'):
#			return self.get_cache('hypernyms')
#
#		print 'calculating hypernym counts...'
#
#		w = nltk.corpus.wordnet
#
#		return_data = []
#		for row in self.data:
#
#			# split on any series of characters that aren't letters, 
#			# numbers or the underscore
#			features = self.NON_ALPHANUM.split(row[1])
#
#			# get the synsets, then get their hypernyms
#			synsets = reduce(lambda x,y: x + w.synsets(y), features, [])
#			hypernyms = reduce(lambda x,y: x + [y] + y.hypernyms(), 
#				synsets, [])
#			wordnet_features = reduce(
#				lambda x,y: x + [l.name for l in y.lemmas], hypernyms, []
#			)
#
#			features = features + wordnet_features
#
#			# add the synsets and hypernyms (by name) to the as-found words
#			features = Counter(features)
#
#			# get rid of empty-string features
#			del features['']
#
#			class_name = row[2]
#			return_data.append((class_name, dict(features)))
#
#		self.cache('hypernyms', return_data)
#
#		return return_data


class VectorList(object):
	def __init__(self, data, word_list):
		self.data = data
		self.word_list = word_list
		self.pointer = 0

	def __iter__(self):
		return self

	def next(self):

		self.pointer += 1

		if len(self.data[self.pointer])==3:
			idx, frequencies, class_name =  self.data[self.pointer-1]
			return (idx, [
				frequencies[w] if w in frequencies else 0 
				for w in self.word_list
			], class_name)

		else:
			idx, frequencies =  self.data[self.pointer-1]
			return (idx, [
				frequencies[w] if w in frequencies else 0 
				for w in self.word_list
			])






import csv
import os
import json
import re

try:
	import nltk
except ImportError:
	pass

from collections import defaultdict, Counter

def sort_by_category(input_fname, output_fname):

	input_fh = open(input_fname, 'r')
	output_fh = open(output_fname, 'w')

	reader = csv.reader(input_fh)

	sorted_data = defaultdict(lambda: [])
	for row in reader:
		features = row[1].split()
		class_name = row[2]
		sorted_data[class_name].append([class_name] + features)

	output_fh.write(json.dumps(sorted_data, indent=2))

	input_fh.close()
	output_fh.close()



class Data(object):

	# some useful regular expressions
	NON_ALPHA = re.compile(r'[^a-zA-Z]+')
	NON_ALPHANUM = re.compile(r'[^a-zA-Z0-9_]+')

	# Where shold things be stored and read?
	FNAMES = {
		'read':'data/raw/train.csv',
		'as_frequencies': 'data/cache/frequencies.json',
		'lemmatized': 'data/cache/lemmatized.json',
		'hypernyms': 'data/cache/hypernyms.json',
		'tficf': 'data/cache/tficf.json',
		'class_counts': 'data/cache/class_counts.json',
		'icf_scores': 'data/cache/icf_scores.json',
	}

	# Keep things in cache so we don't need to keep calculating
	CACHE = {}


	# Some flags
	NOT_CACHED = 0
	FILE_CACHED = 1
	MEM_CACHED = 2


	def __init__(self, limit=None, fname="data/raw/train.csv"):

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
			print 'loading from memory cache'
			return self.CACHE[name]

		if os.path.isfile(self.FNAMES[name]):
			print 'loading from file cache'
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


	def tficf(self, use_cache=True):
		'''
		the features are weighted frequency counts, where the frequency
		count is weighted by inverse class frequency.
		'''

		# check for cached data first
		if use_cache and self.check_cache('tficf'):
			return self.get_cache('tficf')

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

		self.cache('icf_scores', icf_scores)



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

		return_data = []
		for row in self.data:

			# split on any series of characters that aren't letters, 
			# numbers or the underscore
			features = Counter(self.NON_ALPHANUM.split(row[1]))

			# get rid of empty-string features
			del features['']

			class_name = row[2]
			return_data.append((class_name, dict(features)))

		self.cache('as_frequencies', return_data)

		return return_data


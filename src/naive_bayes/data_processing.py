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


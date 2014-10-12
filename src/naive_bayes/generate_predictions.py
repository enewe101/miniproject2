import naive_bayes as nb
import copy
import csv

# add this project's src folder to the path
import sys
import os
sys.path.append(os.path.abspath('../'))

import data_processing as dp


OPTIONS = {
	'representation': 'as_tficf',
	'lemmatize': False,
	'find_specials': False,
	'remove_stops': True,
	'use_digrams': True,
	#'limit': 100
}


CROSS_VAL_FNAME = 'data/results/naive_bayes/NB_cross_val_predictions.csv'
TEST_FNAME = 'data/results/naive_bayes/NB_test_predictions.csv'


def generate_predictions(options):

	options = copy.deepcopy(options)
	representation = options.pop('representation')
	limit = options.pop('limit', None)

	data = dp.Data(limit=limit)

	train_data = getattr(data, representation)(use_cache=False, **options)
	test_data = getattr(data, representation)(data_part='test', **options)

	#generate_cross_val_predictions(train_data)
	generate_test_set_predictions(train_data, test_data)


def generate_cross_val_predictions(data):

	# open a file to receive the prediction results
	fh = open(CROSS_VAL_FNAME, 'w')
	writer = csv.writer(fh)

	# generate predictions in cross validation
	cvt = nb.CrossValTester(data)
	predictions = cvt.generate_crossval_predictions(10)

	# write the predicitons to file
	for prediction in predictions:
		writer.writerow(prediction)

	fh.close()


def generate_test_set_predictions(train_data, test_data):

	# open a file in which predictions for the test set will be written
	fh = open(TEST_FNAME, 'w')
	writer = csv.writer(fh)

	# make a naive bayes classifier and train it on the full training set
	nbc = nb.NaiveBayesClassifier()
	nbc.train(train_data)

	# get the test set predictions
	for datum in test_data:
		idx, features = datum
		prediction = nbc.classify(features)
		writer.writerow((idx, prediction))

	fh.close()


if __name__ == '__main__':
	generate_predictions(OPTIONS)

from knn import CrossValCase
from sim_sweep import sim_sweep

# Total number of training examples is 96213

SCHEDULE = {
	'nn_k' : [10],
	'dist' : ['EUCLID_DIST', 'COSINE_DIST'],
	'representation' : ['as_tficf'],
	'lemmatize' : [True],
	'find_specials' : [True],
	'remove_stops' : [True],
	'use_digrams' : [False],
	'limit' : [96213]
}

FNAME = 'results.json'

def do_cross_validation_test_suite():
	sim_sweep(CrossValCase, FNAME, SCHEDULE)

if __name__ == '__main__':
	do_cross_validation_test_suite()

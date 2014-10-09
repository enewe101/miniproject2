import sys
from naive_bayes import CrossValCase
from sim_sweep import sim_sweep

SCHEDULE = {
	'representation': ['as_frequencies', 'as_tfidf', 'as_tficf'],
	#'lemmatize': [True, False],
	#'find_specials': [True, False],
	#'remove_stops': [True, False],
	#'use_digrams': [True, False],
	#'limit': [1000]
}

FNAME = 'test.json'

def do_cross_validation_test_suite(num_procs=None):
	if num_procs is not None:
		print 'using %d processes.' % num_procs
	else:
		print 'using all available processors.'

	sim_sweep(CrossValCase, FNAME, SCHEDULE, num_procs=num_procs)

if __name__ == '__main__':
	if len(sys.argv)>1:
		num_procs = int(sys.argv[1])
	else:
		num_procs = None

	do_cross_validation_test_suite(num_procs)


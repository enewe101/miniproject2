import sys
from naive_bayes import CrossValCase
from sim_sweep import sim_sweep

SCHEDULE = {
	'representation': ['as_modified_tficf'], 
	'lemmatize': [False],
	'find_specials': [False],
	'remove_stops': [False],
	'use_digrams': [False],
	#'limit': [1000]
}

FNAME = 'tf.json'

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


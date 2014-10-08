from naive_bayes import CrossValCase
from sim_sweep import sim_sweep

SCHEDULE = {
	'representation': ['as_frequencies', 'as_tfidf', 'as_tficf'],
	'lemmatize': [True, False],
	'find_specials': [True, False],
	'remove_stops': [True, False],
	'use_digrams': [True, False],
	'limit': [1000]
}

FNAME = 'test.json'

def do_cross_validation_test_suite():
	sim_sweep(CrossValCase, FNAME, SCHEDULE)

if __name__ == '__main__':
	do_cross_validation_test_suite()


from svm import CrossValCase
from sim_sweep import sim_sweep

# Total number of training examples is 96213

SCHEDULE = {
	'classifier_rep' : ['LinearSVC'],
	'representation' : ['as_tfidf', 'as_tficf'],
	'lemmatize' : [True],
	'find_specials' : [True],
	'remove_stops' : [True],
	'use_digrams' : [False],
	'limit' : [5000]
}

FNAME = 'results.json'

def do_cross_validation_test_suite():
	sim_sweep(CrossValCase, FNAME, SCHEDULE)

if __name__ == '__main__':
	do_cross_validation_test_suite()

from svm import GenPredictionsCase
from sim_sweep import sim_sweep

# Total number of training examples is 96213

SCHEDULE = {
	'classifier_rep' : ['LinearSVC'],
	'representation' : ['as_tfidf', 'as_tficf'],
	'lemmatize' : [True],
	'find_specials' : [True],
	'remove_stops' : [True],
	'use_digrams' : [False],
	'limit' : [1000]
}

FNAME = 'results.json'

def do_gen_preds_test_suite():
	sim_sweep(GenPredictionsCase, FNAME, SCHEDULE)

if __name__ == '__main__':
	do_gen_preds_test_suite()

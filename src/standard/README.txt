README
======

To run 10-fold cross-validation with kNN, just execute the source file knn.py, i.e. `python knn.py` in the console. The output will be the accuracy result of the cross-validation.

Prior to cross-validation, the dataset will be lemmatized, removed of stop words, and special tokens will be binned, after which tf-icf weights will be applied. Just before the weights are applied, those words with tf-icf score less than 1.000001 will be removed from the feature set. (See report for details.)

By default, the cross-validation will be performed on the entire training set, but for quick results, you can provide the number of examples you want to consider in the command-line incovation of the program, e.g. `python knn.py 200` for a training set size of 200.

import sys
sys.path.insert(0, '/home/2014/iabdel1/scikit-learn')

from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import csv
import numpy as np

def int_classes(cls):
    c=0    
    if cls=='cs':
	    c=1
    if cls=='physics':
        c=2
    if cls=='math':
        c=3
    if cls=='stat':
        c=4
    return c

def rewrite_classes(X):
    classes = []
    for elt in X:
        if elt == 1:
            elt = 'cs'
        elif elt == 2:
            elt = 'physics'
        elif elt == 3:
            elt  = 'math'
        elif elt == 4:
            elt = 'stat'
        classes.append(elt)
    return classes

# read files
lda_linear_svc_data_file = open('../git/miniproject2/data/results/lda/linear_svc_train_256.csv')
lda_linear_svc_data_reader = csv.reader(lda_linear_svc_data_file)
lda_linear_svc_data = [row for row in lda_linear_svc_data_reader]
lda_linear_svc_data_file.close()

lda_logistic_regression_data_file = open('../git/miniproject2/data/results/lda/logistic_regression_train_256.csv')
lda_logistic_regression_data_reader = csv.reader(lda_logistic_regression_data_file)
lda_logistic_regression_data = [row for row in lda_logistic_regression_data_reader]
lda_logistic_regression_data_file.close()

lda_gaussian_naive_bayes_data_file = open('../git/miniproject2/data/results/lda/gaussian_naive_bayes_train_256.csv')
lda_gaussian_naive_bayes_data_reader = csv.reader(lda_gaussian_naive_bayes_data_file)
lda_gaussian_naive_bayes_data = [row for row in lda_gaussian_naive_bayes_data_reader]
lda_gaussian_naive_bayes_data_file.close()
"""
lda_random_forest_data_file = open('../git/miniproject2/data/results/lda/random_forest_train_256.csv')
lda_random_forest_data_reader = csv.reader(lda_random_forest_data_file)
lda_random_forest_data = [row for row in lda_random_forest_data_reader]
lda_random_forest_data_file.close()

lda_decision_tree_data_file = open('../git/miniproject2/data/results/lda/decision_tree_train_256.csv')
lda_decision_tree_data_reader = csv.reader(lda_decision_tree_data_file)
lda_decision_tree_data = [row for row in lda_decision_tree_data_reader]
lda_decision_tree_data_file.close()"""

naive_bayes_data_file = open('../git/miniproject2/data/results/naive_bayes/NB_cross_val_predictions.csv')
naive_bayes_data_reader = csv.reader(naive_bayes_data_file)
naive_bayes_data = [row for row in naive_bayes_data_reader]
naive_bayes_data_file.close()

class_file  = open('../clean/train_output.csv')
class_data_reader = csv.reader(class_file)
class_data_reader.next()
class_data = [int_classes(row[1]) for row in class_data_reader]
class_file.close()

# read testing files
lda_linear_svc_test_file = open('../git/miniproject2/data/results/lda/linear_svc_test_256.csv')
lda_linear_svc_test_reader = csv.reader(lda_linear_svc_test_file)
lda_linear_svc_test = [row for row in lda_linear_svc_test_reader]
lda_linear_svc_test_file.close()

lda_logistic_regression_test_file = open('../git/miniproject2/data/results/lda/logistic_regression_test_256.csv')
lda_logistic_regression_test_reader = csv.reader(lda_logistic_regression_test_file)
lda_logistic_regression_test = [row for row in lda_logistic_regression_test_reader]
lda_logistic_regression_test_file.close()

lda_gaussian_naive_bayes_test_file = open('../git/miniproject2/data/results/lda/gaussian_naive_bayes_test_256.csv')
lda_gaussian_naive_bayes_test_reader = csv.reader(lda_gaussian_naive_bayes_test_file)
lda_gaussian_naive_bayes_test = [row for row in lda_gaussian_naive_bayes_test_reader]
lda_gaussian_naive_bayes_test_file.close()
"""
lda_random_forest_test_file = open('../git/miniproject2/data/results/lda/random_forest_test_256.csv')
lda_random_forest_test_reader = csv.reader(lda_random_forest_test_file)
lda_random_forest_test = [row for row in lda_random_forest_test_reader]
lda_random_forest_test_file.close()

lda_decision_tree_test_file = open('../git/miniproject2/data/results/lda/decision_tree_test_256.csv')
lda_decision_tree_test_reader = csv.reader(lda_decision_tree_test_file)
lda_decision_tree_test = [row for row in lda_decision_tree_test_reader]
lda_decision_tree_test_file.close()"""

naive_bayes_test_file = open('../git/miniproject2/data/results/naive_bayes/NB_test_predictions.csv')
naive_bayes_test_reader = csv.reader(naive_bayes_test_file)
naive_bayes_test = [row for row in naive_bayes_test_reader]
naive_bayes_test_file.close()


# combine files
input_data = []

for i in range(len(lda_linear_svc_data)):
    input_data.append([
        int_classes(lda_linear_svc_data[i][1]), 
        int_classes(naive_bayes_data[i][1]),
        int_classes(lda_logistic_regression_data[i][1]),
        int_classes(lda_gaussian_naive_bayes_data[i][1]),
        #int_classes(lda_random_forest_data[i][1]),
        #int_classes(lda_decision_tree_data[i][1])
    ])

test_data = []
for i in range(len(lda_linear_svc_test)):
    test_data.append([
        int_classes(lda_linear_svc_test[i][1]), 
        int_classes(naive_bayes_test[i][1]),
        int_classes(lda_logistic_regression_test[i][1]),
        int_classes(lda_gaussian_naive_bayes_test[i][1]),
        #int_classes(lda_random_forest_test[i][1]),
        #int_classes(lda_decision_tree_test[i][1])
    ])


# convert to np.array
input_data = np.array(input_data)
test_data = np.array(test_data)
class_data = np.array(class_data)

# preprocessing
encoder = OneHotEncoder(sparse=False)
encoder.fit(test_data)
test_data = encoder.transform(test_data)

encoder = OneHotEncoder(sparse=False)
encoder.fit(input_data)
input_data = encoder.transform(input_data)

cv = cross_validation.KFold(len(input_data),n_folds=10,indices=False)

score = 0
print '10-fold cross validation using DecisionTreeClassifier classifier:'
for train,test in cv:
    model = DecisionTreeClassifier()
    model.fit(input_data[train], class_data[train])
    temp = model.score(input_data[test],class_data[test])
    print temp
    score = score + temp
score = score/10
print score



print 'predicting test data using DecisionTreeClassifier classifier:'
model = DecisionTreeClassifier()
model.fit(input_data, class_data)
train_predictions = rewrite_classes(model.predict(input_data))
test_predictions  = rewrite_classes(model.predict(test_data))
f_train = open('../results/ensemble/decision_tree_train.csv','w')
f_test  = open('../results/ensemble/decision_tree_test.csv','w')
csv_train = csv.writer(f_train)
csv_test = csv.writer(f_test)
csv_train.writerows([(i, train_predictions[i]) for i in range(len(train_predictions))])
csv_test.writerows([(i, test_predictions[i]) for i in range(len(test_predictions))])
f_train.close()
f_test.close()
print 'done, files written'


import sys
sys.path.insert(0, '/home/2014/iabdel1/scikit-learn')

from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
import csv
import json
import numpy as np

n_feat = 256

data_file = open('../clean/document-topics-'+str(n_feat)+'-usable.csv')
data_reader = csv.reader(data_file)

test_file = open('../clean/test-topics-'+str(n_feat)+'-usable.csv')
test_reader = csv.reader(test_file)

test_data = [[float(elt) for elt in row] for row in test_reader]

input_data = []
class_data = []

for row in data_reader:
    input_data.append([float(i) for i in row[:-1]] )
    if row[-1]=='cs':
	    c= 1
    if row[-1]=='physics':
        c=2
    if row[-1]=='math':
        c=3
    if row[-1]=='stat':
        c=4
    class_data.append(c)

data_file.close()

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

input_data = np.array(input_data)
class_data = np.array(class_data)

test_data = np.array(test_data)


print 'prediction using Gaussian Naive Bayes classifier:'
model = GaussianNB()
model.fit(input_data, class_data)
train_predictions = rewrite_classes(model.predict(input_data))
test_predictions  = rewrite_classes(model.predict(test_data))
f_train = open('../results/gaussian_naive_bayes_train_'+str(n_feat)+'.csv','w')
f_test  = open('../results/gaussian_naive_bayes_test_'+str(n_feat)+'.csv','w')
csv_train = csv.writer(f_train)
csv_test = csv.writer(f_test)
csv_train.writerows([(i, train_predictions[i]) for i in range(len(train_predictions))])
csv_test.writerows([(i, test_predictions[i]) for i in range(len(test_predictions))])
f_train.close()
f_test.close()
print 'done, files written'

print 'prediction using Multinomial Naive Bayes classifier:'
model = MultinomialNB()
model.fit(input_data, class_data)
train_predictions = rewrite_classes(model.predict(input_data))
test_predictions  = rewrite_classes(model.predict(test_data))
f_train = open('../results/multinomial_naive_bayes_train_'+str(n_feat)+'.csv','w')
f_test  = open('../results/multinomial_naive_bayes_test_'+str(n_feat)+'.csv','w')
csv_train = csv.writer(f_train)
csv_test = csv.writer(f_test)
csv_train.writerows([(i, train_predictions[i]) for i in range(len(train_predictions))])
csv_test.writerows([(i, test_predictions[i]) for i in range(len(test_predictions))])
f_train.close()
f_test.close()
print 'done, files written'

print 'prediction using Random Forest classifier:'
model = RandomForestClassifier()
model.fit(input_data, class_data)
train_predictions = rewrite_classes(model.predict(input_data))
test_predictions  = rewrite_classes(model.predict(test_data))
f_train = open('../results/random_forest_train_'+str(n_feat)+'.csv','w')
f_test  = open('../results/random_forest_test_'+str(n_feat)+'.csv','w')
csv_train = csv.writer(f_train)
csv_test = csv.writer(f_test)
csv_train.writerows([(i, train_predictions[i]) for i in range(len(train_predictions))])
csv_test.writerows([(i, test_predictions[i]) for i in range(len(test_predictions))])
f_train.close()
f_test.close()
print 'done, files written'


print 'prediction using LinearSVC classifier:'
model = svm.LinearSVC()
model.fit(input_data, class_data)
train_predictions = rewrite_classes(model.predict(input_data))
test_predictions  = rewrite_classes(model.predict(test_data))
f_train = open('../results/linear_svc_train_'+str(n_feat)+'.csv','w')
f_test  = open('../results/linear_svc_test_'+str(n_feat)+'.csv','w')
csv_train = csv.writer(f_train)
csv_test = csv.writer(f_test)
csv_train.writerows([(i, train_predictions[i]) for i in range(len(train_predictions))])
csv_test.writerows([(i, test_predictions[i]) for i in range(len(test_predictions))])
f_train.close()
f_test.close()
print 'done, files written'

print 'prediction using DecisionTreeClassifier classifier:'
model = DecisionTreeClassifier()
model.fit(input_data, class_data)
train_predictions = rewrite_classes(model.predict(input_data))
test_predictions  = rewrite_classes(model.predict(test_data))
f_train = open('../results/decision_tree_train_'+str(n_feat)+'.csv','w')
f_test  = open('../results/decision_tree_test_'+str(n_feat)+'.csv','w')
csv_train = csv.writer(f_train)
csv_test = csv.writer(f_test)
csv_train.writerows([(i, train_predictions[i]) for i in range(len(train_predictions))])
csv_test.writerows([(i, test_predictions[i]) for i in range(len(test_predictions))])
f_train.close()
f_test.close()
print 'done, files written'

print 'prediction using LogisticRegression:'
model = LogisticRegression()
model.fit(input_data, class_data)
train_predictions = rewrite_classes(model.predict(input_data))
test_predictions  = rewrite_classes(model.predict(test_data))
f_train = open('../results/logistic_regression_train_'+str(n_feat)+'.csv','w')
f_test  = open('../results/logistic_regression_test_'+str(n_feat)+'.csv','w')
csv_train = csv.writer(f_train)
csv_test = csv.writer(f_test)
csv_train.writerows([(i, train_predictions[i]) for i in range(len(train_predictions))])
csv_test.writerows([(i, test_predictions[i]) for i in range(len(test_predictions))])
f_train.close()
f_test.close()
print 'done, files written'


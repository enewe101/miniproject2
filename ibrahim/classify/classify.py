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

input_data = np.array(input_data)
class_data = np.array(class_data)

score = 0
scores = {}

cv = cross_validation.KFold(len(input_data),n_folds=10,indices=False)

print '10-fold cross validation using Gaussian Naive Bayes classifier:'
for train,test in cv:
    model = GaussianNB()
    model.fit(input_data[train], class_data[train])
    score = score +  model.score(input_data[test],class_data[test])
score = score/10
print score

scores['naive_bayes'] = score

score = 0
print '10-fold cross validation using Multinomial Naive Bayes classifier:'
for train,test in cv:
    model = MultinomialNB()
    model.fit(input_data[train], class_data[train])
    score = score +  model.score(input_data[test],class_data[test])
score = score/10
print score

scores['multinomial_naive_bayes'] = score

score = 0
print '10-fold cross validation using Random Forest classifier:'
for train,test in cv:
    model = RandomForestClassifier()
    model.fit(input_data[train], class_data[train])
    score = score +  model.score(input_data[test],class_data[test])
score = score/10
print score

scores['random_forest'] = score

score = 0
print '10-fold cross validation using LinearSVC classifier:'
for train,test in cv:
    model = svm.LinearSVC()
    model.fit(input_data[train], class_data[train])
    score = score +  model.score(input_data[test],class_data[test])
score = score/10
print score

scores['linear_svc'] = score

score = 0
print '10-fold cross validation using DecisionTreeClassifier classifier:'
for train,test in cv:
    model = DecisionTreeClassifier()
    model.fit(input_data[train], class_data[train])
    score = score +  model.score(input_data[test],class_data[test])
score = score/10
print score

scores['decision_tree'] = score

score = 0
print '10-fold cross validation using LogisticRegression:'
for train,test in cv:
    #poly = PolynomialFeatures(interaction_only=True,include_bias=False)
    #poly_train = poly.fit_transform(input_data[train])
    #poly_test = poly.fit_transform(input_data[test])
    model = LogisticRegression()
    model.fit(input_data[train], class_data[train])
    score = score +  model.score(input_data[test],class_data[test])
score = score/10
print score

scores['logistic_regression'] = score

results = open('results'+str(n_feat),'w')
results.write(json.dumps(scores))
results.close()

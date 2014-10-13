from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import csv
import numpy as np


data_file = open('../clean/document-topics-16-usable.csv')
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

cv = cross_validation.KFold(len(input_data),k=10,indices=False)

print '10-fold cross validation using Gaussian Naive Bayes classifier:'
for train,test in cv:
    model = GaussianNB()
    model.fit(input_data[train], class_data[train])
    print model.score(input_data[test],class_data[test])

print '10-fold cross validation using Multinomial Naive Bayes classifier:'
for train,test in cv:
    model = MultinomialNB()
    model.fit(input_data[train], class_data[train])
    print model.score(input_data[test],class_data[test])


print '10-fold cross validation using LinearSVC classifier:'
for train,test in cv:
    model = svm.LinearSVC()
    model.fit(input_data[train], class_data[train])
    print model.score(input_data[test],class_data[test])

print '10-fold cross validation using DecisionTreeClassifier classifier:'
for train,test in cv:
    model = DecisionTreeClassifier()
    model.fit(input_data[train], class_data[train])
    print model.score(input_data[test],class_data[test])

print '10-fold cross validation using LogisticRegression:'
for train,test in cv:
    model = LogisticRegression()
    model.fit(input_data[train], class_data[train])
    print model.score(input_data[test],class_data[test])


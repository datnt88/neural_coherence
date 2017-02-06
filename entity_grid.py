from __future__ import absolute_import
from __future__ import division

import pickle
from six.moves import cPickle
import gzip
import random
import numpy as np

import glob, os, csv, re
from collections import Counter
import itertools
import collections

from sklearn import svm, linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

def get_transition(line="", w_size=2):
	x = line.split()
	x = x[1:]
	e_occur = x.count('X') + x.count('S') + x.count('O') #couting the number of occurence for the entities for salince

	pre = '1'
	if e_occur ==2 :
		pre = '2'
	elif e_occur == 3:
		pre = '3'
	elif e_occur >3 :
		pre = '4'
                
	new_x = []
	len_ = len(x) # check if len =2
	#print(len_)
	#if len_ == 2:

	for i, role in enumerate(x):
		if i < len_ -1:
			next_role = x[i+1]

			if role == '-':
				if next_role == '-':
					new_x.append('-Y-Y')
				else:
					new_x.append('-Y' + next_role + pre)
			else:
				if next_role == '-':
					new_x.append(role + pre +'-Y')
				else:
					new_x.append(role +pre+ next_role + pre)

	#print(x)
	#print(new_x)					
	#print(len(new_x))

	return new_x



#get feature names
def get_feature(w_size=2):
	vocabs=['S1', 'O1', 'X1', 'S2', 'O2', 'X2', 'S3', 'O3', 'X3', 'S4', 'O4', 'X4', '-Y']
	products = list(itertools.product(vocabs,repeat=w_size))
	feats = []
	for f_ in products:
		feats.append("".join(f_))	
	#print(len(feats))
	return feats


def get_feature_scores(filename="", salience=3, w_size=2, feats=['SSS']):
	lines = [line.rstrip('\n') for line in open(filename)]

	counts = []
	for line in lines:
		trans = get_transition(line=line)
		#print(trans)
		counts = counts +  trans
		
	total_count = len(counts)
	
	counts = dict(collections.Counter(counts))
	#print(counts)

	f_scores = []
	for f_ in feats:
		ocur = counts.get(f_) #using salinece here
		if ocur == None:
			ocur = 0.0
		f_scores.append(ocur/total_count)		
		
	#print(f_scores)
	return f_scores

def check_feats(feat_1,feat_0):
	n = len(feat_1)
	for i in range(0,n):
		if feat_1[i] != feat_0[i]:
			return 1 
	return -1


def loading_data(filelist="", perm_num=20, w_size=2):

	feats = get_feature(w_size=w_size)
	#print(feats)

	list_of_files = [line.rstrip('\n') for line in open(filelist)]
	
	X_1 = []
	X_0 = []

	for file in list_of_files:
		#print(file)
		feat_1 = get_feature_scores(filename=file + ".EGrid",feats=feats)

		p_count = 0
		for i in range(1, perm_num + 1):
			feat_0 = get_feature_scores(filename=file + ".EGrid" + "-" + str(i), feats=feats)
			#check if there is duplication
			if check_feats(feat_0, feat_1) != -1:
				p_count = p_count + 1
				X_0.append(feat_0)

		for i in range (0, p_count): #stupid code
                        X_1.append(feat_1)
	
	assert len(X_0) == len(X_1)
	#print(X_0)
	return X_1, X_0

print("----------------------------------------------------")
print("Loading training data...!")
X_train_1, X_train_0 = loading_data(filelist="final_data/list.train_dev.0001.docs")
n_sample = len(X_train_1)
print("Number of train: " + str(n_sample))
np.random.seed(113)
np.random.shuffle(X_train_1)
np.random.seed(113)
np.random.shuffle(X_train_0)


Xp, yp = [],[]
#n_sample = 10

k = int(n_sample/2);
for i in range(0,k):
	Xp.append(np.array(X_train_1[i]) - np.array(X_train_0[i]))
	yp.append(1)

for i in range(k,n_sample):
	Xp.append(np.array(X_train_0[i]) - np.array(X_train_1[i]))
	yp.append(0)

assert len(Xp) == n_sample


np.random.seed(2016)
np.random.shuffle(Xp)
np.random.seed(2016)
np.random.shuffle(yp)


print("Start to traing the model...!")
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100]},
                    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100], 'gamma': [1e-3, 1e-4]}]

#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1]},
#                    {'kernel': ['linear'], 'C': [0.1, 1], 'gamma': [1e-4]}]



clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10, scoring='precision_macro')
clf.fit(Xp, yp) 

print("Best parameters set found on development set:")
print(clf.best_params_)
print("\n")
print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
	print(" --> %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
	

#pickle.dump(clf, open( "clf_c1.p", "wb" ))
print("Training with gridSearch done...!")

print("Loading test data...!")
X_test_1, X_test_0 = loading_data(filelist="final_data/list.test.docs.final")
n_test = len(X_test_1)
print("Number of test: " + str(n_test))
Xp_test = []
for i in range(0,n_test):
	Xp_test.append(np.array(X_test_1[i]) - np.array(X_test_0[i]))
	

assert len(Xp_test) == n_test

y_predict = clf.predict(Xp_test)
y_predict = list(y_predict)

total = 20411 
wins = y_predict.count(1)
loss = y_predict.count(0)
ties = total - n_test


print(' - Wins: ' +  str(wins))
print(' - Loss: ' +  str(loss))
print(' - Ties: ' +  str(ties))

print(' - Accuracy: ' +  str(wins/total))


#print(len(X_test_1))














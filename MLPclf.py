#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:53:01 2017

@author: yjt
"""

from sklearn.neural_network import MLPClassifier
import numpy as np


data =[]
labels =[]
with open("train.txt") as ifile:
        for line in ifile:
            tokens = line.strip('\n').split(' ')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
Xtrain = np.array(data)
ytrain = np.array(labels)

data =[]
labels =[]
with open("test.txt") as ifile:
        for line in ifile:
            tokens = line.strip('\n').split(' ')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
Xtest = np.array(data)
ytest = np.array(labels)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                   hidden_layer_sizes=(5, 2), random_state=1)

y_pred = clf.fit(Xtest, ytest).predict(Xtest)
print("Number of mislabeled points out of a total %d points : %d"
       % (Xtest.shape[0],(ytest != y_pred).sum()))
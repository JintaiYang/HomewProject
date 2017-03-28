#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:47:01 2017

@author: yjt
"""

import numpy as np
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from dataprocessing import plotimg


def plot_decision_function(classifier,axis,min_x,max_x,min_y, max_y,title):
    xx,yy = np.meshgrid(np.linspace(min_x-2, max_x+2,500),np.linspace(min_y-2, max_y+2,500))

    Z = classifier.decision_function(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    axis.contourf(xx,yy,Z,c=ytrain,alpha = 0.75,cmap = plt.cm.bone)
    color = ['r','b']
    axis.scatter(Xtrain[:,0],Xtrain[:,1],c=color,alpha = 0.9,
                 cmap = plt.cm.bone)
    axis.axis('on')

    axis.set_title(title)

def plot_hyperplane(classifier, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = classifier.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x , max_x )  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


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


clf = LogisticRegression()

clf.fit(Xtrain,ytrain)

min_x = np.min(Xtrain[:, 0])
max_x = np.max(Xtrain[:, 0])

min_y = np.min(Xtrain[:, 1])
max_y = np.max(Xtrain[:, 1])



fig,axes = plt.subplots(1,figsize = (10,10))
plot_decision_function(clf,axes,min_x,max_x,min_y, max_y,'Constant weights')
plot_hyperplane(clf, min_x, max_x, 'k--','clf')


plt.xlim(min_x-1,max_x+1)
plt.ylim(min_y-1,max_y+1)


scores=cross_val_score(clf,Xtest,ytest,cv=5)
print '准确率',np.mean(scores),scores

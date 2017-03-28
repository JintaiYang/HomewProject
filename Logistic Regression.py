#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:09:02 2017

@author: yjt
"""
from sklearn.linear_model.logistic import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
#from dataprocessing import plotimg,readfile
import dataprocessing as dp


rfile =dp.readfile()
Xtrain,ytrain = rfile.readfile('train.txt')
Xtest,ytest = rfile.readfile('test.txt')


clf = LogisticRegression(C=1e5)
clf.fit(Xtrain,ytrain)

pltimg =dp.plotimg(clf,Xtrain,ytrain)

fig,axes = plt.subplots(1,figsize = (8,6))

pltimg.plot_classlpane(axes,'Constant weights')
pltimg.plot_hyperplane('k--','clf')



scores=cross_val_score(clf,Xtest,ytest,cv=5)
print '准确率',scores.mean()









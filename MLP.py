#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 21:03:34 2017

@author: yjt
"""

from sklearn.neural_network import MLPClassifier
from dataprocessing import readfile,plotimg
import matplotlib.pyplot as plt


rfile =readfile()
Xtrain,ytrain = rfile.readfile('train.txt')
Xtest,ytest = rfile.readfile('test.txt')



clf = MLPClassifier(alpha = 1e-5)
clf.fit(Xtrain,ytrain)

pltimg =plotimg(clf,Xtrain,ytrain)


fig,axes = plt.subplots(1,figsize = (8,6))

pltimg.plot_classlpane(axes,'Constant weights')




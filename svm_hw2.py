#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:35:53 2017

@author: yjt
"""

from sklearn.svm import SVC
import matplotlib.pyplot as plt
from dataprocessing import readfile,plotimg


rfile =readfile()
Xtrain,ytrain = rfile.readfile('train.txt')
Xtest,ytest = rfile.readfile('test.txt')




clf =SVC(kernel='linear',C=1e1)
clf.fit(Xtrain,ytrain)

pltimg =plotimg(clf,Xtrain,ytrain)


fig,axes = plt.subplots(1,figsize = (8,6))

pltimg.plot_classlpane(axes,'Constant weights')














#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:52:21 2017

@author: yjt
"""

from sklearn import naive_bayes as NB
from dataprocessing import readfile,plotimg
import matplotlib.pyplot as plt



rfile =readfile()
Xtrain,ytrain = rfile.readfile('train.txt')
Xtest,ytest = rfile.readfile('test.txt')




gnb = NB.GaussianNB()
bnb = NB.BernoulliNB()
mnb = NB.MultinomialNB()

gnb.fit(Xtrain,ytrain)
bnb.fit(Xtrain,ytrain)



gnb_pltimg =plotimg(gnb,Xtrain,ytrain)
bnb_pltimg =plotimg(gnb,Xtrain,ytrain)

figure = plt.figure(figsize=(8, 4))

axes = plt.subplot(1,2,1)
gnb_pltimg.plot_classlpane(axes,'Constant weights')
axes = plt.subplot(1,2,2)
bnb_pltimg.plot_classlpane(axes,'Constant weights')



y_pred = gnb.predict(Xtrain)
print("Number of mislabeled points out of a total %d points : %d"
       % (Xtrain.shape[0],(ytrain != y_pred).sum()))

y_pred = bnb.fit(Xtrain, ytrain).predict(Xtrain)
print("Number of mislabeled points out of a total %d points : %d"
       % (Xtrain.shape[0],(ytrain != y_pred).sum()))





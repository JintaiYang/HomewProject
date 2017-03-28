#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:05:56 2017

@author: yjt
"""

import numpy as np
import matplotlib.pyplot as plt

class plotimg():

    def __init__(self,clf,Xtrain,ytrain):

        self.min_x = np.min(Xtrain[:, 0])
        self.max_x = np.max(Xtrain[:, 0])
        self.min_y = np.min(Xtrain[:, 1])
        self.max_y = np.max(Xtrain[:, 1])
        self.classifier = clf
        self.Xtrain=Xtrain
        self.ytrain = ytrain

    def plot_classlpane(self,axis,title):
        xx,yy = np.meshgrid(np.linspace(self.min_x-2, self.max_x+2,500),np.linspace(self.min_y-2, self.max_y+2,500))

        Z = self.classifier.predict(np.c_[xx.ravel(),yy.ravel()])
        Z = Z.reshape(xx.shape)

        axis.contourf(xx,yy,Z,c=self.ytrain,alpha = 0.75,cmap = plt.cm.bone)
        color = ['r','b']
        axis.scatter(self.Xtrain[:,0],self.Xtrain[:,1],c=color,alpha = 0.9,
                     cmap = plt.cm.bone)
        axis.axis('on')
        axis.set_title(title)

    def plot_hyperplane(self, linestyle, label):
        # get the separating hyperplane
        w = self.classifier.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(self.min_x , self.max_x )  # make sure the line is long enough
        yy = a * xx - (self.classifier.intercept_[0]) / w[1]
        plt.plot(xx, yy, linestyle, label=label)

        plt.xlim(self.min_x-1,self.max_x+1)
        plt.ylim(self.min_y-1,self.max_y+1)

class readfile():

    def readfile(self,filename):
        data=[]
        labels=[]
        with open(filename) as ifile:
            for line in ifile:
                tokens = line.strip('\n').split(' ')
                data.append([float(tk) for tk in tokens[:-1]])
                labels.append(tokens[-1])
                Xtrain = np.array(data)
                ytrain = np.array(labels)
        return Xtrain,ytrain






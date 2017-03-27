#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 22:28:39 2017

@author: yjt
"""
import numpy as np
import scipy as sp
#from sklearn import svm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC



def plot_decision_function(classifier,axis,min_x,max_x,min_y, max_y,title):
    xx,yy = np.meshgrid(np.linspace(min_x-2, max_x+2,500),np.linspace(min_y-2, max_y+2,500))

    Z = classifier.decision_function(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)

    axis.contourf(xx,yy,Z,alpha = 0.75,cmap = plt.cm.bone)
    axis.scatter(X[:,0],X[:,1],c=y,alpha = 0.9,
                 cmap = plt.cm.bone)
    axis.axis('on')

    axis.set_title(title)

def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x , max_x )  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


data   = []
labels = []
with open("train.txt") as ifile:
        for line in ifile:
            tokens = line.strip('\n').split(' ')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
X = np.array(data)
y = np.array(labels)



pltx = X[:,0]
plty = X[:,1]


clf = SVC(kernel='rbf')
clf.fit(X, y)

#plt.plot(clf.decision_function(X))

#print(clf.predict([[0.8, 1]]))

plt.scatter(pltx,plty,c=y)


suppv=clf.support_vectors_;

suppvxx=suppv[0:len(suppv),0]
suppvyy=suppv[0:len(suppv),1]

min_x = np.min(X[:, 0])
max_x = np.max(X[:, 0])

min_y = np.min(X[:, 1])
max_y = np.max(X[:, 1])



fig,axes = plt.subplots(1,figsize = (10,10))
plot_decision_function(clf,axes,min_x,max_x,min_y, max_y,'Constant weights')

#plot_hyperplane(clf, min_x, max_x, 'k--', 'Boundary\nfor clf ')


plt.scatter(suppvxx,suppvyy)

plt.xlim(min_x-1,max_x+1)
plt.ylim(min_y-1,max_y+1)

plt.show()
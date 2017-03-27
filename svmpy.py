#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:35:53 2017

@author: yjt
"""

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def plot_decision_function(classifier,axis,title):
    xx,yy = np.meshgrid(np.linspace(-4,5,500),np.linspace(-4,5,500))

    Z = classifier.decision_function(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)

    axis.contourf(xx,yy,Z,alpha = 0.75,cmap = plt.cm.bone)
    axis.scatter(X[:,0],X[:,1],c=y,alpha = 0.9,
                 cmap = plt.cm.bone)
    axis.axis('off')
    axis.set_title(title)

def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


'''
x=np.random.rand(20,2);
x1=np.vstack((np.ones(x[0:10][:].shape),np.zeros(x[0:10][:].shape)))
x1=x1.reshape(x.shape)
X = x+x1;
y = np.hstack((np.ones(len(X[0:10][:])),np.zeros(len(X[0:10][:]))))
'''

X=np.array([[2,2],[4,4],[4,0],[0,0],[2,0],[0,2]])
y=np.array([1,1,1,-1,-1,-1])

#pltx = [x[0] for x in X]
#plty = [x[1] for x in X]


pltx = X[0:6,0]
plty = X[0:6,1]


clf = SVC(kernel='linear')
clf.fit(X, y)

plt.plot(clf.decision_function(X))

print(clf.predict([[0.8, 1]]))

plt.scatter(pltx,plty,c=y)

fig,axes = plt.subplots(1,figsize = (10,10))
plot_decision_function(clf,axes,'Constant weights')
suppv=clf.support_vectors_;

suppvxx=suppv[0:len(suppv[:,0]),0]
suppvyy=suppv[0:len(suppv[:,0]),1]

min_x = np.min(X[:, 0])
max_x = np.max(X[:, 0])

min_y = np.min(X[:, 1])
max_y = np.max(X[:, 1])

plot_hyperplane(clf, min_x, max_x, 'r--', 'Boundary\nfor clf ')

plt.scatter(suppvxx,suppvyy)
plt.show()
















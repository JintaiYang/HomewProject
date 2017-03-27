#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 18:24:32 2017

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


X = np.array([[2, 2], [4, 4], [4, 0], [0, 0],[2,0],[0,2]])
y = np.array([1, 1, 1, -1,-1,-1])

pltx = [x[0] for x in X]
plty = [x[1] for x in X]

clf = SVC()
clf.fit(X, y)


plt.plot(clf.decision_function(X))

print(clf.predict([[0.8, 1]]))

plt.scatter(pltx,plty,c=y)

fig,axes = plt.subplots(1,2,figsize = (14,6))
plot_decision_function(clf,axes[0],'Constant weights')
plt.show()




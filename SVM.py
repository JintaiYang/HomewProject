#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 16:42:18 2017

@author: yjt
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cross_validation import cross_val_score

def plot_decision_function(classifier,sample_weight,axis,title):
    xx,yy = np.meshgrid(np.linspace(-4,5,500),np.linspace(-4,5,500))

    Z = classifier.decision_function(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)

    axis.contourf(xx,yy,Z,alpha = 0.75,cmap = plt.cm.bone)
    axis.scatter(X[:,0],X[:,1],c=y,s=100*sample_weight,alpha = 0.9,
                 cmap = plt.cm.bone)
    axis.axis('off')
    axis.set_title(title)





np.random.seed(0)
X = np.r_[np.random.randn(10,2)+[1,1],np.random.randn(10,2)]
y = [1]*10+[-1]*10

sample_weight_last_ten = abs(np.random.randn(len(X)))
sample_weight_constant = np.ones(len(X))
sample_weight_last_ten[15:] *= 5
sample_weight_last_ten[9] *= 15


clf_weights = svm.SVC(kernel= 'linear')
clf_weights.fit(X,y,sample_weight = sample_weight_last_ten)
clf_no_weights = svm.SVC(kernel= 'linear')
clf_no_weights.fit(X,y)

fig,axes = plt.subplots(1,2,figsize = (14,6))
plot_decision_function(clf_no_weights,sample_weight_constant,axes[0],'Constant weights')
plot_decision_function(clf_weights,sample_weight_last_ten,axes[1],'Modified weights')
plt.show()


scores=cross_val_score(clf_weights,X,y,cv=5)
print '准确率',np.mean(scores),scores







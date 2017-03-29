#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================
Created on Sun Mar 26 22:28:39 2017

@author: yjt

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import dataprocessing as dp
from sklearn.linear_model.logistic import LogisticRegression

h = .02  # step size in the mesh

names = [#"Linear SVM",
         "LogisticRegression",
         "RBF SVM","MPL","Naive Bayes"]

classifiers = [
    #SVC(kernel="linear", C=1),
    LogisticRegression(C=1),
    SVC(gamma=2, C=1),
    MLPClassifier(alpha=1),
    GaussianNB(),
    ]


rfile =dp.readfile()
X_train,y_train = rfile.readfile('train.txt')
X_test,y_test = rfile.readfile('test.txt')


datasets = [(X_train, y_test)]
figure = plt.figure(figsize=(20, 6))
X, y = (X_test, y_test)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                 np.arange(y_min, y_max, h))

i=1
# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['r', 'b'])

ax = plt.subplot(len(datasets), len(classifiers)+1,i)
ax.set_title("TestPoint")

# Plot the training points and testing points
#ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

for name, clf in zip(names, classifiers):
    i += 1;
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the test points
    #ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')

plt.tight_layout()
plt.show()

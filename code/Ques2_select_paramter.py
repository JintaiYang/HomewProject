#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 21:17:50 2017

@author: yjt
"""

from __future__ import print_function

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.linear_model.logistic import LogisticRegression
from dataprocessing import readfile
import matplotlib.pyplot as plt
import dataprocessing as dp


#read data
rfile =readfile()
X_train,y_train = rfile.readfile('train.txt')
X_test,y_test = rfile.readfile('test.txt')


# Set the parameters by cross-validation
#'''
#for SVM
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,1e-1,1e0,1e1,1e2],
                     'C': [0.1,1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.1,1, 10, 100, 1000]}]

'''
#for LogisticRegression
tuned_parameters = [{'penalty': ['l1'],
                     'C': [1e-1, 1, 10, 100, 1000]},
                    {'penalty': ['l2'], 'C': [1e-1, 1, 10, 100, 1000]}]

'''
scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    '''
    clf = GridSearchCV(LogisticRegression(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    '''
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)

    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.


pltimg =dp.plotimg(clf.best_estimator_,X_test,y_test)
fig,axes = plt.subplots(1,figsize = (6,4))
#pltimg.plot_classlpane(axes,'The Best parameter of LR classifier')
pltimg.plot_classlpane(axes,'The Best parameter of SVC classifier')
#pltimg.plot_hyperplane('k--','clf')





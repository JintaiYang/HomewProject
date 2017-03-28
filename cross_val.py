#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:22:42 2017

@author: yjt
"""
'''
from sklearn import preprocessing
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0)

iris = datasets.load_iris()
scaler = preprocessing.StandardScaler().fit(X_train)

X_train_transformed = scaler.transform(X_train)

clf = svm.SVC(C=1).fit(X_train_transformed, y_train)

X_test_transformed = scaler.transform(X_test)
score=clf.score(X_test_transformed, y_test)

'''


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score


data = pd.read_csv('Advertising.csv', index_col=0)
feature_cols = ['TV', 'Radio', 'Newspaper']

X = data[feature_cols]
y = data.Sales

lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')
print scores



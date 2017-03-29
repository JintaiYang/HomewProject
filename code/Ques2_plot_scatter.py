#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:09:02 2017

@author: yjt
"""
import matplotlib.pyplot as plt
import dataprocessing as dp
from matplotlib.colors import ListedColormap


rfile =dp.readfile()
Xtrain,ytrain = rfile.readfile('train.txt')
Xtest,ytest = rfile.readfile('test.txt')

cm = plt.cm.RdBu
cm_bright = ListedColormap(['r', 'b'])

fig,ax = plt.subplots(1,2,figsize = (8,6))

ax[0].scatter(Xtrain[:,0],Xtrain[:,1],c=ytrain,cmap=cm_bright,alpha = 0.9)
ax[0].set_title('train ponit')


ax[1].scatter(Xtest[:,0],Xtest[:,1],c=ytest,cmap=cm_bright,alpha = 0.9)
ax[1].set_title('test ponit')








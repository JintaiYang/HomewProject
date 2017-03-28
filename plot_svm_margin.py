#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
the 4th Homework of  MachineLearn
author: Jintai Yang
school number 201621010609
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

X = np.array([[2, 2], [4, 4], [4, 0], [0, 0],[2,0],[0,2]])
Y = np.array([1, 1, 1, -1,-1,-1])
fignum =1

Penalty=np.array([0.05,0.1,0.2,0.3,0.5,0.8,1,100])

plt.figure(figsize=(10, 6))
for penalty in Penalty:
    clf = svm.SVC(kernel='linear', C=penalty)
    clf.fit(X, Y)

    x_min = min(X[:,0])-1
    x_max = max(X[:,0])+1
    y_min = min(X[:,1])-1
    y_max = max(X[:,1])+1

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(x_min, x_max)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # separating hyperplane
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))

    #print margin
    yy_down = yy + a * margin
    yy_up = yy - a * margin

    plt.subplot(2,4,fignum)

    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    color = ['r']*3+['b']*3
    plt.scatter(X[:, 0], X[:, 1], c=color, zorder=10, cmap=plt.cm.Paired)

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], c='g',s=80,
               zorder=10)

    plt.axis('tight')
    plt.title('penalty=' +str(penalty))



    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

 # Put the result into a color plot
    Z = Z.reshape(XX.shape)

   # plt.figure(fignum, figsize=(4, 3))

    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())

    fignum = fignum + 1
plt.subplots_adjust(left=.1, right=0.95, bottom=0.15, top=0.95)
plt.savefig('scatterfig.tif', dpi=100)
plt.show()

plt.figure(figsize=(6,4))
color = ['r']*3+['b']*3
plt.scatter(X[:, 0], X[:, 1], c=color, zorder=10, cmap=plt.cm.Paired)
plt.axis('tight')
plt.savefig('1.tif',dpi=100)


print




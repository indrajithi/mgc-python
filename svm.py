# File: svm.py
# Author: Indrajith Indraptrastham
# Date: Sun Apr 30 23:23:52 IST 2017


import numpy as np 
import acc
import feature
from sklearn import svm
from sklearn.model_selection import cross_val_score
import random

#load pre saved variables
X = np.load('train70.npy')
Y = np.load('trainY70.npy')
xt = np.load('test30.npy')
yt = np.load('testY30.npy')
Xall = np.load('allX.npy')
Yall = np.load('allY.npy')

def svms():
    #linear kernel=======================================================
    print('='*5+' Linear Kernel SVM '+'='*5)
    clf = svm.LinearSVC(C=10000)
    clf.fit(X,Y)

    #cross validation
    resub = clf.predict(X)
    resubAcc = acc.get(resub,Y)
    #test case
    print('Resubstituion Accuracy : ',resubAcc)
    resTest = clf.predict(xt)
    tesAcc = acc.get(resTest,yt)
    print('Test case accuracy : ',tesAcc)
    scores = cross_val_score(clf, X, Y, cv=5)
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print('='*30)

    #Polynomial kernel=======================================================
    print('='*5+' Polynomial Kernel SVM '+'='*5)
    clf = svm.SVC(kernel='poly',C=1000)
    clf.fit(X,Y)

    #cross validation
    resub = clf.predict(X)
    resubAcc = acc.get(resub,Y)
    #test case
    print('Resubstituion Accuracy : ',resubAcc)
    resTest = clf.predict(xt)
    tesAcc = acc.get(resTest,yt)
    print('Test case accuracy : ',tesAcc)

    scores = cross_val_score(clf, xt, yt, cv=5)
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print('='*30)


def cross_validation(train_percentage,fold)
""" Accepts parameter: 
        train_percentage;
        fold;
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 07:12:40 2021

@author: josef
"""
import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

class leastSquaresClassifier:
    #one-vs-all classifier using least squared loss function
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)
    
class logLinearClassifier:
    #one-vs-all classifier using logistic loss function
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True
    def funObj(self, w, X, y):
        yXw = y * X.dot(w)
    
        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))
    
        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)
    
        return f, g
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons

            #utils.check_gradient(self, X, ytmp)
            (self.W[i], f) = findMin.findMin(self.funObj, self.W[i],
                                          self.maxEvals, X, ytmp, verbose=self.verbose)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)
    
class softmaxClassifier:
    #Using softmax function, single classifier could output multi class result
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True
        
    def funObj(self, w, X, y):
        n, d = X.shape
        k=self.n_classes
        W=w.reshape(k,d)
        g=np.zeros([k,d])
        I=np.zeros(len(y))
        f=0
        # Calculate the function value
        for i in range(n):
            f+=-X[i].dot(W[y[i]])+np.log(np.sum(np.exp(W.dot(X[i]))))
            
        # Calculate the gradient value
        for c in range(k):
            prob=self.cal_prob(c, W, X)
            Itmp=I.copy()
            Itmp[y==c]=1
            g[c]=X.T.dot(prob-Itmp)
        
        g=g.flatten()
        return f, g
    
    def cal_prob(self,c,W,X):
        n, d = X.shape
        p=np.zeros(n)
        for i in range(n):
            p[i]=np.exp(W[c].dot(X[i]))/np.sum(np.exp(W.dot(X[i])))
        return p
            
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.w = np.zeros((self.n_classes*d))


        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)

        self.W=self.w.reshape(self.n_classes,d)
        
    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)


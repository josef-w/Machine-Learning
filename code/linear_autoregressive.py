#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 04:27:00 2021

@author: josef
"""
import numpy as np
from numpy.linalg import solve
from scipy.optimize import approx_fprime
import utils
import math

class LeastSquaresAutoregressive:

    def fit(self,X,k,hyper=None):
        self.k = k
        self.t = X.size
        X_new = self.autoregression(X)
        y = X[(k-1):]
        size_v=self.t-self.k+1
        #creat weight matrix V. V is identity matrix if you don't want to give weight to features. 
        V=np.eye(size_v)
        if hyper is not None:
            for i in range(0,size_v):
                if i<=(size_v-hyper[0]):
                    V[i,i]=hyper[2]
                elif i<=(size_v-hyper[1]):
                    V[i,i]=hyper[3]
                else:
                    V[i,i]=hyper[4]

        self.w = solve(X_new.T@V@X_new, X_new.T@V@y)



    def predict(self, X):
        y = np.copy(X[(self.t-self.k):])
        y[0] = 1
        return y.T@self.w

    def autoregression(self, X):
        T = X.size
        K = self.k
        Z = np.ones((T-K+1,K))
        for i in range(T-K+1):
            X_curr = X[i:(K-1+i)]
            X_curr = X_curr.reshape((1,X_curr.size))[0]
            Z[i,1:] = X_curr
        return Z

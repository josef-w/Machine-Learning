#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 04:42:00 2021

@author: josef
"""
import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    """Ordinary Least Squares"""
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    """Least squares where each sample point X has a weight associated with it."""
    def fit(self,X,y,V):
        self.w=solve(X.T@V@X, X.T@V@y)
        pass

class LinearModelGradient(LeastSquares):
    
    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d,1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w.reshape(d,), lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):

        # Calculate the function value
        #f = 0.5*np.sum((X@w - y)**2)
        f=np.sum(np.log(np.exp(X@w - y)+np.exp(y-X@w)))
        # Calculate the gradient value
        #g = X.T@(X@w-y)
        g = np.zeros((1,1))
        for i in range(0,500):
            g+=(X[i]@(w.T@X[i]-y[i])-X[i]@(y[i]-w.T@X[i]))/(np.exp(w.T@X[i]-y[i])+np.exp(y[i]-w.T@X[i]))
        return f,g


# Least Squares with a bias added
class LeastSquaresBias:
    """Least Squares with a bias added"""
    def fit(self,X,y):
        bias=np.ones(np.size(X,0))
        X=np.insert(X, 0, values=bias, axis=1)
        self.w = solve(X.T@X, X.T@y)
        pass

    def predict(self, X):
        bias=np.ones(np.size(X,0))
        X=np.insert(X, 0, values=bias, axis=1)
        return X@self.w

# Least Squares with polynomial basis
class LeastSquaresPoly:
    """Least Squares with polynomial basis"""
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        self.leastSquares.fit(self.__polyBasis(X),y)
        #print(self.leastSquares.w)
    def predict(self, X):
        
        return self.leastSquares.predict(self.__polyBasis(X))

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        Z=np.ones((np.size(X,0),1))
        if self.p >0:
            for i in range(1, self.p+1):
                Z=np.c_[Z,X**i]
     
        return Z


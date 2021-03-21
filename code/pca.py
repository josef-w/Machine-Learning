#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:10:56 2021

@author: josef
"""
import numpy as np
from findMin import findMin

class PCA:
    '''
    Solves the PCA problem min_Z,W (Z*W-X)^2 using SVD
    '''

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.mu = np.mean(X,axis=0)
        X = X - self.mu

        U, s, Vh = np.linalg.svd(X)
        self.W = Vh[:self.k]

    def compress(self, X):
        X = X - self.mu
        Z = X@self.W.T
        return Z

    def expand(self, Z):
        X = Z@self.W + self.mu
        return X

class AlternativePCA(PCA):
    """
    Solves the PCA problem min_Z,W (Z*W-X)^2 using gradient descent
    """
    def fit(self, X):
        n,d = X.shape
        k = self.k
        self.mu = np.mean(X,0)
        X = X - self.mu

        # Randomly initial Z, W
        z = np.random.randn(n*k)
        w = np.random.randn(k*d)

        for i in range(10): # do 10 "outer loop" iterations
            z, f = findMin(self._fun_obj_z, z, 10, w, X, k)
            w, f = findMin(self._fun_obj_w, w, 10, z, X, k)
            print('Iteration %d, loss = %.1f' % (i, f))

        self.W = w.reshape(k,d)

    def compress(self, X):
        n,d = X.shape
        k = self.k
        X = X - self.mu
        # We didn't enforce that W was orthogonal 
        # so we need to optimize to find Z
        # (or do some matrix operations)
        z = np.zeros(n*k)
        z,f = findMin(self._fun_obj_z, z, 100, self.W.flatten(), X, k)
        Z = z.reshape(n,k)
        self.Z=Z
        return Z

    def _fun_obj_z(self, z, w, X, k):
        n,d = X.shape
        Z = z.reshape(n,k)
        W = w.reshape(k,d)

        R = np.dot(Z,W) - X
        f = np.sum(R**2)/2
        g = np.dot(R, W.transpose())
        return f, g.flatten()

    def _fun_obj_w(self, w, z, X, k):
        n,d = X.shape
        Z = z.reshape(n,k)
        W = w.reshape(k,d)

        R = np.dot(Z,W) - X
        f = np.sum(R**2)/2
        g = np.dot(Z.transpose(), R)
        return f, g.flatten()

class RobustPCA(AlternativePCA):
    def _fun_obj_z(self, z, w, X, k):
        n,d = X.shape
        Z = z.reshape(n,k)
        W = w.reshape(k,d)

        Epsilon=0.0001*np.ones([n,d])
        R = np.dot(Z,W) - X
        f = np.sum(np.sqrt(R**2+Epsilon))
        g = np.zeros([n,k])
        for i in range(n):
            for j in range(k):
                g[i,j]=0.5*np.dot(np.true_divide(R[i],np.sqrt(R[i]**2+Epsilon[i])),W[j])
        return f, g.flatten()

    def _fun_obj_w(self, w, z, X, k):
        n,d = X.shape
        Z = z.reshape(n,k)
        W = w.reshape(k,d)
        Epsilon=0.0001*np.ones([n,d])

        R = np.dot(Z,W) - X
        f = np.sum(np.sqrt(R**2+Epsilon))
        #g = np.dot(Z.transpose(), R)
        g = np.zeros([k,d])
        for i in range(k):
            for j in range(d):
                g[i,j]=0.5*np.dot(np.true_divide(R[:,j],np.sqrt(R[:,j]**2+Epsilon[:,j])),Z[:,i])
        
        return f, g.flatten()

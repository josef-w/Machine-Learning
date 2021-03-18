#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 04:33:46 2021

@author: josef
"""

import numpy as np
from scipy import stats
import utils
from numpy.linalg import norm

class KNN:
    """
    Implementation of k-nearest neighbours classifier
    """
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y

    def predict(self, Xtest):
        X = self.X
        y = self.y
        n = X.shape[0]
        t = Xtest.shape[0]
        k = min(self.k, n)

        # Compute cosine_distance distances between X and Xtest
        dist2 = self.cosine_distance(X, Xtest)

        # yhat is a vector of size t with integer elements
        yhat = np.ones(t, dtype=np.uint8)
        for i in range(t):
            # sort the distances to other points
            inds = np.argsort(dist2[:,i])

            # compute mode of k closest training pts
            yhat[i] = stats.mode(y[inds[:k]])[0][0]

        return yhat

    def euclidean_dist_squared(X1, X2):
        """Computes the Euclidean distance between rows of 'X1' and rows of 'X2'
    
        Parameters
        ----------
        X1: an N by D numpy array
        X2: an T by D numpy array
    
        Returns: an array of size N by T containing the pairwise squared Euclidean distances.
        It exploits the identity (a-b)^2 = a^2 + b^2 - 2ab.
        The right-hand-side of the above is more amenable to vector/matrix operations.
        """
        return np.sum(X1**2, axis=1)[:,None] + np.sum(X2**2, axis=1)[None] - 2 * np.dot(X1,X2.T)
    def cosine_distance(X1,X2):
        """
        Computes the cosine distance between rows of 'X1' and rows of 'X2'
    
        Parameters
        ----------
        X1: an N by D numpy array
        X2: an T by D numpy array
    
        Returns: an array of size N by T containing the pairwise cosine distances.
        if there is any zero row in X1 or X2, set the distance between any zero row in X1 to all the rows X2 to zero or vice verca.
        """
        N=X1.shape[0]
        T=X2.shape[0]
        norm1 = norm(X1,axis=1).reshape(X1.shape[0],1)
        norm2 = norm(X2,axis=1).reshape(1,X2.shape[0])
        end_norm = np.dot(norm1,norm2)
        ###set a positive value to the rows and columns in end_norm that equals to 0, 
        ###so that the final result matrix will be 0  on these places.
        end_norm[norm1[:,0]==0,:]=1
        end_norm[:,norm2[0]==0]=1
        cos=np.zeros([N,T])
        cos = np.dot(X1, X2.T)/end_norm
        return cos

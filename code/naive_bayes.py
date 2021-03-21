#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 04:32:16 2021

@author: josef
"""
import numpy as np
import math

class NaiveBayes:

    def __init__(self, num_classes, beta=0):
        self.num_classes = num_classes
        self.beta = beta

    def fit(self, X, y):
        N, D = X.shape

        # Compute the number of class labels
        C = self.num_classes
        
        pi=math.pi
        # Compute the probability of each class i.e p(y==c)
        counts = np.bincount(y.astype(np.int32))
        p_y = counts / N
        logp_y= np.log(p_y)
        # Compute the conditional probabilities i.e.
        # p(x(i,j)=1 | y(i)==c) as p_xy
        # p(x(i,j)=0 | y(i)==c) as p_xy
        u1=np.mean(X[y==1], axis=0)
        u0=np.mean(X[y==0], axis=0)
        u=np.append([u0],[u1],axis=0)
        o1_squ=np.mean(np.power(u1-X[y==1],2),axis=0)
        o0_squ=np.mean(np.power(u0-X[y==0],2),axis=0)
        o_squ=np.append([o0_squ],[o1_squ],axis=0)
        logp_xy = np.zeros((N, C)) 
        
        for c in range(C):
            for n in range(N):
                divid=np.true_divide(X[n]-u[c],np.sqrt(o_squ[c]))
                logp_xy[n, c] = -np.sum(0.5*np.power(divid, 2)+0.5*np.log(o_squ[c]*2*math.pi))


        self.logp_y = logp_y
        self.logp_xy = logp_xy
        self.u=u
        self.o_squ=o_squ



    def predict(self, X):

        N, D = X.shape
        C = self.num_classes
        logp_xy = np.zeros((N, C)) 
        logp_y = self.logp_y
        u=self.u
        o_squ=self.o_squ

        y_pred = np.zeros(N)
        for n in range(N):
            for c in range(C):
                divid=np.true_divide(X[n]-u[c],np.sqrt(o_squ[c]))
                logp_xy[n, c] = -np.sum(0.5*np.power(divid, 2)+0.5*np.log(o_squ[c]*2*math.pi))

            logprobs = logp_y.copy() # initialize with the p(y) terms
            logprobs=logp_xy[n]+logprobs
            y_pred[n] = np.argmax(logprobs)

        return y_pred


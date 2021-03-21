#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 05:08:42 2021

@author: josef
"""
from random_stump import *
from decision_tree import *
import numpy as np

class RandomTree(DecisionTree):
        
    def __init__(self, max_depth):
        DecisionTree.__init__(self, max_depth=max_depth, stump_class=RandomStumpGiniIndex)

    def fit(self, X, y, thresholds=None):
        N = X.shape[0]
        boostrap_inds = np.random.choice(N, N, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]
        
        DecisionTree.fit(self, bootstrap_X, bootstrap_y)

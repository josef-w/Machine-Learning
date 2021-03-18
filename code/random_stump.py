#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 05:06:12 2021

@author: josef
"""
import numpy as np
import utils
from decision_stump import *


class RandomStumpInfoGain(DecisionStumpInfoGain):
    
    def fit(self, X, y):
        
        # Randomly select k features.
        # This can be done by randomly permuting
        # the feature indices and taking the first k
        D = X.shape[1]
        k = int(np.floor(np.sqrt(D)))
        
        chosen_features = np.random.choice(D, k, replace=False)
                
        DecisionStumpInfoGain.fit(self, X, y, split_features=chosen_features)

class RandomStumpGiniIndex(DecisionStumpGiniIndex):

        def fit(self, X, y, thresholds=None):
            # Randomly select k features.
            # This can be done by randomly permuting
            # the feature indices and taking the first k
            D = X.shape[1]
            k = int(np.floor(np.sqrt(D)))

            chosen_features = np.random.choice(D, k, replace=False)

            DecisionStumpGiniIndex.fit(self, X, y, split_features=chosen_features, thresholds=thresholds)



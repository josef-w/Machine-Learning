#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:36:17 2021

@author: josef
"""
# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# our code
import utils

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question
    np.seterr(divide='ignore', invalid='ignore')
    
    if question == "1":
        #import our code
        from decision_tree import *
        from random_tree import *
        from random_forest import *
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X,y = dataset["X"], dataset["y"]
        Xtest, ytest = dataset["Xtest"], dataset["ytest"] 

        #model = RandomTree(max_depth=6)
        #model = RandomForest(num_trees=3,max_depth=6)
        model=DecisionTree(max_depth=6, stump_class=DecisionStumpInfoGain)
        
        time1=time.time()
        model.fit(X, y)
        utils.evaluate_model(model,X,y,Xtest,ytest)
        time2=time.time()
        print("Time:%.3f"%(time2-time1))
        
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "decisionBoundary.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
    elif question == '2':
        #our code
        from knn import KNN
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']
        
        model=KNN(k=4)
        utils.evaluate_model(model,X,y,Xtest,ytest)
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "knn.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
    elif question == '3':
        from kmeans import Kmeans
        X = load_dataset('clusterData.pkl')['X']
        ###make multiple initializations
        for iterate in range(0,50):
            model_temp = Kmeans(k=4)
            model_temp.fit(X)
            y = model_temp.predict(X)
            SSE=model_temp.error(X)
            if iterate ==0:
                error_opt=SSE
                model=model_temp
            elif SSE<error_opt:
                error_opt=SSE
                model=model_temp
        y=model.predict(X)
        print(model.error(X))
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
    elif question == "4":
        #our code
        import least_squared_regression
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = least_squared_regression.LinearModelGradient()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.png")
    elif question == '5': 
        #our code
        from pca import *
        X = load_dataset('highway.pkl')['X'].astype(float)/255
        n,d = X.shape
        print(n,d)
        h,w = 64,64      # height and width of each image

        k = 5            # number of PCs
        threshold = 0.1  # threshold for being considered "foreground"

        model = AlternativePCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_pca = model.expand(Z)
        
        model = RobustPCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_robust = model.expand(Z)

        fig, ax = plt.subplots(2,3)
        for i in range(10):
            ax[0,0].set_title('$X$')
            ax[0,0].imshow(X[i].reshape(h,w).T, cmap='gray')

            ax[0,1].set_title('$\hat{X}$ (L2)')
            ax[0,1].imshow(Xhat_pca[i].reshape(h,w).T, cmap='gray')
            
            ax[0,2].set_title('$|x_i-\hat{x_i}|$>threshold (L2)')
            ax[0,2].imshow((np.abs(X[i] - Xhat_pca[i])<threshold).reshape(h,w).T, cmap='gray', vmin=0, vmax=1)

            ax[1,0].set_title('$X$')
            ax[1,0].imshow(X[i].reshape(h,w).T, cmap='gray')
            
            ax[1,1].set_title('$\hat{X}$ (L1)')
            ax[1,1].imshow(Xhat_robust[i].reshape(h,w).T, cmap='gray')

            ax[1,2].set_title('$|x_i-\hat{x_i}|$>threshold (L1)')
            ax[1,2].imshow((np.abs(X[i] - Xhat_robust[i])<threshold).reshape(h,w).T, cmap='gray', vmin=0, vmax=1)

            utils.savefig('highway_{:03d}.jpg'.format(i))
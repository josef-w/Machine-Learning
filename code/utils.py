#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 04:23:37 2021

@author: josef
"""
import pickle
import os
import sys
import numpy as np
from scipy.optimize import approx_fprime
from numpy.linalg import norm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix as sparse_matrix
    

def standardize_cols(X, mu=None, sigma=None):
    """Standardize each column with mean 0 and variance 1"""
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma, mu, sigma


def check_gradient(model, X, y):
    # This checks that the gradient implementation is correct
    w = np.random.rand(model.w.size)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(w,
                                       lambda w: model.funObj(w,X,y)[0],
                                       epsilon=1e-6)

    implemented_gradient = model.funObj(w, X, y)[1]

    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
        raise Exception('User and numerical derivatives differ:\n%s\n%s' %
             (estimated_gradient[:5], implemented_gradient[:5]))
    else:
        print('User and numerical derivatives agree.')


def classification_error(y, yhat):
    return np.mean(y!=yhat)

def evaluate_model(model,X,y,X_test,y_test):
  model.fit(X,y)

  y_pred = model.predict(X)
  tr_error = np.mean(y_pred != y)

  y_pred = model.predict(X_test)
  te_error = np.mean(y_pred != y_test)
  print("    Training error: %.3f" % tr_error)
  print("    Testing error: %.3f" % te_error)

def test_and_plot(model,X,y,Xtest=None,ytest=None,title=None,filename=None):
    """Used for regression;Assume the input model has been fit"""
    #Compute training error
    yhat = model.predict(X)
    trainError = np.mean((yhat - y)**2)
    print("Training error = %.1f" % trainError)

    # Compute test error
    if Xtest is not None and ytest is not None:
        yhat = model.predict(Xtest)
        testError = np.mean((yhat - ytest)**2)
        print("Test error     = %.1f" % testError)

    # Plot model
    plt.figure()
    plt.plot(X,y,'b.')

    # Choose points to evaluate the function
    Xgrid = np.linspace(np.min(X),np.max(X),1000)[:,None]
    ygrid = model.predict(Xgrid)
    plt.plot(Xgrid, ygrid, 'g')

    if title is not None:
        plt.title(title)

    if filename is not None:
        filename = os.path.join("..", "figs", filename)
        print("Saving", filename)
        plt.savefig(filename)
    if Xtest is not None and ytest is not None:
        return trainError, testError
    else:
        return trainError
def plotClassifier(model, X, y):
    """plots the decision boundary of the model and the scatterpoints
       of the target values 'y'.

    Assumptions
    -----------
    y : it should contain two classes: '1' and '2'

    Parameters
    ----------
    model : the trained model which has the predict function

    X : the N by D feature array

    y : the N element vector corresponding to the target values

    """
    x1 = X[:, 0]
    x2 = X[:, 1]

    x1_min, x1_max = int(x1.min()) - 1, int(x1.max()) + 1
    x2_min, x2_max = int(x2.min()) - 1, int(x2.max()) + 1

    x1_line =  np.linspace(x1_min, x1_max, 200)
    x2_line =  np.linspace(x2_min, x2_max, 200)

    x1_mesh, x2_mesh = np.meshgrid(x1_line, x2_line)

    mesh_data = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]

    y_pred = model.predict(mesh_data)
    y_pred = np.reshape(y_pred, x1_mesh.shape)

    plt.figure()
    plt.xlim([x1_mesh.min(), x1_mesh.max()])
    plt.ylim([x2_mesh.min(), x2_mesh.max()])

    plt.contourf(x1_mesh, x2_mesh, -y_pred.astype(int), # unsigned int causes problems with negative sign... o_O
                cmap=plt.cm.RdBu, alpha=0.6)

    plt.scatter(x1[y==0], x2[y==0], color="b", label="class 0")
    plt.scatter(x1[y==1], x2[y==1], color="r", label="class 1")
    plt.legend()
def mode(y):
    """Computes the element with the maximum count

    Parameters
    ----------
    y : an input numpy array

    Returns
    -------
    y_mode :
        Returns the element with the maximum count
    """
    if len(y)==0:
        return -1
    else:
        return stats.mode(y.flatten())[0][0]


def euclidean_dist_squared(X, Xtest):
    """Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'

    Parameters
    ----------
    X : an N by D numpy array
    Xtest: an T by D numpy array

    Returns: an array of size N by T containing the pairwise squared Euclidean distances.

    Python/Numpy (and other numerical languages like Matlab and R)
    can be slow at executing operations in `for' loops, but allows extremely-fast
    hardware-dependent vector and matrix operations. By taking advantage of SIMD registers and
    multiple cores (and faster matrix-multiplication algorithms), vector and matrix operations in
    Numpy will often be several times faster than if you implemented them yourself in a fast
    language like C. The following code will form a matrix containing the squared Euclidean
    distances between all training and test points. If the output is stored in D, then
    element D[i,j] gives the squared Euclidean distance between training point
    i and testing point j. It exploits the identity (a-b)^2 = a^2 + b^2 - 2ab.
    The right-hand-side of the above is more amenable to vector/matrix operations.
    """

    return np.sum(X**2, axis=1)[:,None] + np.sum(Xtest**2, axis=1)[None] - 2 * np.dot(X,Xtest.T)

    # without broadcasting:
    # n,d = X.shape
    # t,d = Xtest.shape
    # D = X**2@np.ones((d,t)) + np.ones((n,d))@(Xtest.T)**2 - 2*X@Xtest.T
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

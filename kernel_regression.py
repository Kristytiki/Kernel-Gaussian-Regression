#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
 Author: Zheqi Wu
 Date : 19/04/2018
 Description: This script implements kernel Ridge
 using Gaussian kernel
 
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics.pairwise import euclidean_distances


def Ker_Ridge(X_train,Y_train,lambda_,gamma_):
    """
    Perform kernel ridge regression of Y on X.
    X: an n x p matrix of explanatory variables.
    Y: an n vector of dependent variables. Y can also be a 
    matrix, as long as the function works.
    lambda_: regularization parameter (lambda >= 0)
    gamma_: hyperparameter of Guassian kernel
    
    Returns C
    """    
    n = X_train.shape[0]
    ## rbf kernel
    k = np.reshape([0 for x in xrange(n*n)],(n,n))
    k = np.exp(-gamma_*euclidean_distances(X_train, X_train, squared=True))
    C = np.dot(np.linalg.inv((k+np.diag(np.full(n,lambda_)))),Y_train)    
    
    return C


def predict_Ridge(X_train,Y_train,X_test,Y_test,lambda_,gamma_):
    
    
    C = Ker_Ridge(X_train,Y_train,lambda_,gamma_)
    ker = np.exp(-gamma_*euclidean_distances(X_train, X_test, squared=True))
    f_x = np.dot(ker.T,C)
    
    return f_x

n=150
p=100
lambda_ = 0.001
x1 = np.random.uniform(0,1,n)
x2 = np.random.uniform(0,1,n)
X = np.c_[x1,x2]
Y = np.reshape((x1**2)+(x2**2),(n,1)) + np.random.randn(n,1)
gamma_=2
 ## split data into training and testing part
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


## try different gamma and lambda
# lambda_=1;gamma_=0.01
def plot1(X_train,Y_train,X_test,Y_test,lambda_,gamma_):
    
    f_x=predict_Ridge(X_train,Y_train,X_test,Y_test,lambda_,gamma_)
    fig, ax = plt.subplots()
    ax.plot(np.arange(0,30,1), Y_test, 'r', label='Y')
    s1 = 'hat Y lambda:'
    ax.plot(np.arange(0,30,1), f_x, 'b', label=(s1,lambda_,',gamma:',gamma_))
    legend = ax.legend(loc=1, shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('#00FFCC')
    plt.plot(np.arange(0,30,1),Y_test,'r',f_x,'b')   
    
    return 0
gamma_=np.array([0.01,1,10,1000])
lambda_ = np.array([0.01,1,10,1000])
for i in range (0,4):
    for j in range (0,4):
        plot1(X_train,Y_train,X_test,Y_test,lambda_[i],gamma_[j])
        
plot1(X_train,Y_train,X_test,Y_test,lambda_,gamma_)
# lambda=1,gamma=1
f_x=predict_Ridge(X_train,Y_train,X_test,Y_test,lambda_=1,gamma_=1)
fig, ax = plt.subplots()
ax.plot(np.arange(0,30,1), Y_test, 'r', label='Y')
ax.plot(np.arange(0,30,1), f_x, 'b', label='hat Y(lambda=1,gamma=0.01)')

legend = ax.legend(loc=1, shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('#00FFCC')
#test_error_.argmin(0) 
plt.plot(np.arange(0,30,1),Y_test,'r',f_x,'b')   

plt.show()


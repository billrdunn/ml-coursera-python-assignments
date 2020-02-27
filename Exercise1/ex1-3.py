# 3 Linear regression with multiple variables
import numpy as np
import os
import utils
grader = utils.Grader()
import sys
sys.path.insert(1, '/home/bill/google-drive-grive2/Uni/ML/ml-coursera-python-assignments/Exercise1/Functions')

""" Load text data and turn into matrix """
data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:, :2] # all the rows and up to the third column
y = data[:, 2] # all the rows and the third column
m = y.size # m = number of training sets

""" Print some data points """
# Defines how strings are displayed (eg. alignment and width)
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
# Creates 26 dashes
print('-'*26)
# Print first 10 rows of table
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))

from Functions.featureNormalize import featureNormalize

X_norm, mu, sigma = featureNormalize(X)

grader[4] = featureNormalize

# Add intercept term to X_norm
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

from Functions.computeCostMulti import computeCostMulti
grader[5] = computeCostMulti
from Functions.gradientDescent import gradientDescent
grader[6] = gradientDescent
#grader.grade()

""" NORMAL EQUATIONS """
# the closed-form solution to linear regression is the normal equation
# this formula does not require feature scaling, and no iterations

# Load data
data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis=1)

from Functions.normalEqn import normalEqn
theta = normalEqn(X,y)
print(theta)

grader[7] = normalEqn
grader.grade()





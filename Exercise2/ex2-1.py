# Programming Exercise 2: Logistic Regression
# token: gYwHeBaRoKzsHhtL


import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
import utils
grader = utils.Grader()
import sys
from functions import *

# Load data
# First two columns contain the two exam scores
# Third column contains the binary label (admitted or not)
# add data file to path and load into data, specifying comma delimiter
data = np.loadtxt(os.path.join('Data', 'ex2data1.txt'), delimiter=',')
# set X data as all the rows, columns 0 and 1
# set y data as all the rows, column 2
X, y = data[:, :2], data[:, 2]

plotData(X,y)

grader[1] = sigmoid

# m = 100 (number of training examples), n = 2 (number of predictors)
m, n = X.shape

# add intercept term to X (column of 1s)
X = np.concatenate([np.ones([m,1]), X], axis=1)

initial_theta = np.array([0,0,0])
cost, grad = costFunction(initial_theta, X, y)
print(cost)
print(grad)

grader[2] = costFunction
grader[3] = costFunction

# now use scipy.optimize.minimize to learn parameters which takes inputs:
# costFunction
# initial_theta
# (X, y)
# jac (bool): whether the jacobian (gradient) is returned
# method: optimization algorithm to use
# options (eg. max iterations)

# do the optimisation
options = {'maxiter': 400}

res = optimize.minimize(costFunction, initial_theta, (X,y), jac=True, method='TNC', options=options)
# the fun property returns the value of costFunction at optimised theta
cost = res.fun
# the optimised theta is the x property
theta = res.x
utils.plotDecisionBoundary(plotData, theta, X, y)

# predict admission probability for scores 45 and 85
scores = [1, 45, 85]
prob = sigmoid(np.dot(theta, scores))
print(prob)

# test model on training set
p = predict(theta, X)
print(p)

# find prediction accuracy
acc = np.mean(p==y)
print(acc)
# ie. classifier can correctly classify 89% of training data examples

grader[4] = predict
grader.grade()

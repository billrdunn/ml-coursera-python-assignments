# usual imports
# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils

# define the submission/grader object for this exercise
grader = utils.Grader()

# 20x20 Input Images of Digits
input_layer_size  = 400

# 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
num_labels = 10

#  training data stored in arrays X, y
# ravel "flattens" a matrix into 1D array
data = loadmat(os.path.join('Exercise3/Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in 
# MATLAB where there is no index 0
y[y == 10] = 0

m = y.size

# randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

utils.displayData(sel)

# test values for the parameters theta
theta_t = np.array([-2,-1,1,2], dtype=float)

# test values for the inputs
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)
# print(X_t)

# test values for the labels
y_t = np.array([1, 0, 1, 0, 1])

# test value for the regularisation parameter
lambda_t = 3

from lrCostFunction import lrCostFunction
# test
J_t, grad_t = lrCostFunction(theta_t, X_t, y_t, lambda_t)
# print(J_t)
# print(grad_t)

grader[1] = lrCostFunction

# 1.4 One-vs-all classification
from oneVsAll import oneVsAll
all_theta = oneVsAll(X, y, num_labels, 0.1)
print(all_theta)
grader[2] = oneVsAll
#grader.grade()
# used for manipulating directory paths
import os

# import the numpy module using an alias for the namespace
# eg. np.array([1,2,3]) instead of numpy.array([1,2,3])
import numpy as np

# import the module pyplot from the library matplotlib
# note often written 'import matplotlib.pyplot as plt'
from matplotlib import pyplot
# from mpl_toolkits.mplot3d import Axes3D

# library written for this exercise providing additional functions for assignment submission, and others
import utils

# define the submission/grader object for this exercise
grader = utils.Grader()


# NOTES / TIPS
# always use numpy arrays as python lists do not support vector operations
# print dimensions of numpy arrays using shape property
# by default, numpy does element-wise operations. for matrix multiplication use 'dot' function
# eg. np.dot(A,B)

# define a function which returns a 5x5 identity matrix
def warmUpExercise():
    A = np.eye(5)
    return A


print(warmUpExercise())

grader[1] = warmUpExercise

# load csv data
data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
X, y = data[:, 0], data[:, 1]
m = y.size

# add column of ones to X. The numpy function stack joins arrays along a given axis
X = np.stack([np.ones(m),X], axis=1)

# set number of training examples
m = y.size




# plotData(X,y)

# test cost function with theta = [0.0, 0.0]
from functions import *
J_test1 = computeCost(X,y,[0.0,0.0])
J_test2 = computeCost(X,y,[-1,2])


print(J_test1)
print(J_test2)

grader[2] = computeCost

# test the gradient descent function
theta = np.zeros(2)
iterations = 1500
alpha = 0.01
theta, J_history = gradientDescent(X,y,theta,alpha,iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')

# plot the linear fit
plotData(X[:,1],y)
pyplot.plot(X[:,1], np.dot(X,theta), '-')
pyplot.legend(['Training data', 'Linear regression'])
pyplot.show()

grader[3] = gradientDescent
grader.grade()






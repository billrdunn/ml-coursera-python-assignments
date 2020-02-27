# used for manipulating directory paths
import os

# import the numpy module using an alias for the namespace
# eg. np.array([1,2,3]) instead of numpy.array([1,2,3])
import numpy as np

# import the module pyplot from the library matplotlib
# note often written 'import matplotlib.pyplot as plt'
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# library written for this exercise providing additional functions for assignment submission, and others
import utils

# define the submission/grader object for this exercise
grader = utils.Grader()

import sys
sys.path.insert(1, '/home/bill/google-drive-grive2/Uni/ML/ml-coursera-python-assignments/Exercise1/Functions')


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

from Functions.computeCost import computeCost
# test cost function with theta = [0.0, 0.0]
J_test1 = computeCost(X,y,[0.0,0.0])
J_test2 = computeCost(X,y,[-1,2])


print(J_test1)
print(J_test2)

grader[2] = computeCost

from Functions.gradientDescent import gradientDescent
# test the gradient descent function
theta = np.zeros(2)
iterations = 1500
alpha = 0.01
theta, J_history = gradientDescent(X,y,theta,alpha,iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')

from Functions.plotData import plotData
# plot the linear fit
plotData(X[:,1],y)
pyplot.plot(X[:,1], np.dot(X,theta), '-')
pyplot.legend(['Training data', 'Linear regression'])
#pyplot.show()

grader[3] = gradientDescent
#grader.grade()

# Predict values for population sizes of 35,000 and 70,000
prediction1 = np.dot([1,3.5],theta)
print('For population = 35,000 we predict a profit of {:.2f}\n'.format(prediction1*10000))

prediction2 = np.dot([1,7],theta)
print('For population = 70,000 we predict a profit of {:.2f}\n'.format(prediction2*10000))

# grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialise J_vals to a matrix of 0s
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

# fill out J_vals
for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        J_vals[i,j] = computeCost(X,y, [theta0, theta1])

# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T

# Create a surface plot
fig = pyplot.figure(figsize=(12,5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.title('Surface')

# Create a contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = pyplot.subplot(122)
pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
pyplot.title('Contour, showing minimum')
pyplot.show()

#grader.grade()







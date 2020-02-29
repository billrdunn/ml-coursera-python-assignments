import numpy as np
from matplotlib import pyplot


def plotData(X, y):
# find indicies where y = 0 or 1
    pos = y == 1
    neg = y == 0
    # vector of grades from column 0 for those who were admitted
    X0_1 = X[pos,0]
    # vector of grades from column 1 for those who were admitted
    X1_1 = X[pos,1]
    # vector of grades from column 0 for those who were not admitted
    X0_0 = X[neg,0]
    # vector of grades from column 1 for those who were not admitted
    X1_0 = X[neg,1]

    # create new figure
    pyplot.figure()
    # plot those who were admitted 
    pyplot.plot(X0_1, X1_1, 'k*')
    # plot those who were not admitted
    pyplot.plot(X0_0, X1_0, 'ko')
    # add axes labels
    pyplot.xlabel('Exam 1 score')
    pyplot.ylabel('Exam 2 score')
    pyplot.legend(['Admitted', 'Not admitted'])


def sigmoid(z):
    # Compute sigmoid function given input z
    g = (1 + np.exp(-z))**(-1)
    return g

def costFunction(theta, X, y):

    # find hypothesis function 
    # note no need to transpose with np.dot
    # h is an array of length m
    # note that this h is different to that used in linear regression
    h = sigmoid(np.dot(X, theta))

    J = ( -np.dot(y, np.log(h)) - np.dot( (1-y), np.log(1-h)) ) / y.size

    grad = np.zeros(theta.size)
    for i in range(3):
        grad[i] = np.dot( ( h - y ), X[:,i] ) / y.size

    return J, grad

def predict(theta, X):

    p = sigmoid(np.dot(theta, X.T))
    p = np.round(p)
    return p
    
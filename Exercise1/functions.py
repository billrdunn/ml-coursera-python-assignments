import numpy as np
from matplotlib import pyplot


def plotData(x,y):
    # open a new figure
    # fig = pyplot.figure()
    # ro = red circles, ms = marker size, mec = marker edge colour
    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    pyplot.xlabel('Profit in $10,000')
    pyplot.ylabel('Population of city in 10,000s')




def computeCost(X,y,theta):


    m = y.size

    # add additional column to X data
    J = sum( (np.dot(X,theta) - y)**2 ) / (2 * m)
   
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    # Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    # gradient steps with learning rate `alpha`.

    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    
    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()
    
    J_history = [] # Use a python list to save cost in every iteration
    
    for i in range(num_iters):
        # MATLAB: theta = theta - (alpha / m) * ((X * theta - y)'*X)';
        theta = theta - (alpha / m) * (np.dot((np.dot(X,theta) - y) , X))
        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))
    
    return theta, J_history

def  featureNormalize(X):

    # this is important - something to do with referencing
    X_norm = X.copy()

    # find mean over axis 0
    mu = np.mean(X, axis=0) 
    X_norm = X - mu

    # find stdev 
    sigma = np.std(X_norm, axis=0, ddof=1)
    
    X_norm = X_norm / sigma

    return X_norm, mu, sigma

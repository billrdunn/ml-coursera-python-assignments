import numpy as np
from computeCost import computeCost

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
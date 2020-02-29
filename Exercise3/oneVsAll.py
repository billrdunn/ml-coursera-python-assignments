import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils
from lrCostFunction import lrCostFunction

def oneVsAll(X, y, num_labels, lambda_):
    m, n = X.shape

    # set ititial theta
    initial_theta = np.zeros(n+1)

    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # set options for minimize
    options = {'maxiter' : 50}

    # Run minimize to obtain optimal theta. 
    # This function will return a class object 
    for c in range(num_labels):
        res = optimize.minimize(lrCostFunction, initial_theta, (X, (y==c), lambda_), jac=True, method='TNC', options=options)

    all_theta=res.x

    return all_theta

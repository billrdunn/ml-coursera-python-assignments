import numpy as np
from numpy.linalg import inv, multi_dot

def normalEqn(X, y):

    theta = np.zeros(X.shape[1])

    theta = multi_dot([inv(np.dot(X.T, X)), X.T, y])
 
    return theta
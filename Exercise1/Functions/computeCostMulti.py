import numpy as np

def computeCostMulti(X,y,theta):


    m = y.size

    # add additional column to X data
    J = sum( (np.dot(X,theta) - y)**2 ) / (2 * m)
   
    return J
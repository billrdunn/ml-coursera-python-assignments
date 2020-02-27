import numpy as np

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
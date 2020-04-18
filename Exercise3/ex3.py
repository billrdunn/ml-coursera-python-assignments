import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
from Exercise3 import utils

grader = utils.Grader()

# 20 x 20 input images of digits
input_layer_size = 400

# 10 labels, from 1 to 10
num_labels = 10

# training data stored in arrays X, y
data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()

# map all labels '10' to labels '0' (artefact from matlab)
y[y == 10] = 0

# define number of training examples (m = 5000)
m = y.size

# 1.2 VISUALISING DATA
# randomly select 100 data points to display
rand_indicies = np.random.choice(m, 100, replace=False)
sel = X[rand_indicies, :]  # all the columns in the rand_indicies rows

utils.displayData(sel, False)

# 1.3 VECTORISING LOGISTIC REGRESSION
# define custom data to test vectorised LR
# test values for parameters theta
theta_t = np.array([-2, -1, 1, 2], dtype=float)

# test values for the inputs
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F') / 10.0], axis=1)
print(X_t)

# test values for the labels
y_t = np.array([1, 0, 1, 0, 1])

# test value for the regularisation parameter
lambda_t = 3



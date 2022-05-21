import numpy as np
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()

# nnfs.init() does 3 things:
# 1. sets random seed to 0
# 2. sets dtype to float32
# 3. overrides the original np.dot() product


import matplotlib.pyplot as plt

X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()

# Note: Keep in mind that the neural network will not be aware 
#   of the color differences as the data have no class encodings.

# Each dot is a feature (input)

# Each dot's coordinates are the samples that form the dataset.

# The classification (which we want to calculate with our NN) is 
#   the spiral to which is belongs, i.e., its color

# We can represent each classification (red, blue, green) with
#   a class number (0, 1, 2) for the model to fit
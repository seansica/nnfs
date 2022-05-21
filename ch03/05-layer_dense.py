import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:

	def __init__(self, n_inputs, n_neurons):
		# Init weights and biases
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
		
	def forward(self, inputs):
		# Calc output values from inputs, weights, and biases
		self.output = np.dot(inputs, self.weights) + self.biases
		
# Start testing

# Create a test dataset
X, y = spiral_data(samples=100, classes=3)

# X is a 2-dimensional array, 100x2, i.e. 100 samples of 2-point coordinates

# y refers to a 1-dimensional array of 100 elements where each element can 
# either be 0, 1, or 2, referring to the class of any given sample

# print(X)
# print(y)

# Create Dense Layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Print the output of the first few samples
print(dense1.output[:5])


# Example output: 5 rows of data that has 3 values each; each of 
# the 3 values are the output of the 3 neurons from our dense1 layer

# [[ 0.0000000e+00  0.0000000e+00  0.0000000e+00]
#  [-1.0475188e-04  1.1395361e-04 -4.7983500e-05]
#  [-2.7414842e-04  3.1729150e-04 -8.6921798e-05]
#  [-4.2188365e-04  5.2666257e-04 -5.5912682e-05]
#  [-5.7707680e-04  7.1401405e-04 -8.9430439e-05]]
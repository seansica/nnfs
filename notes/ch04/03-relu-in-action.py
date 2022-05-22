#!/usr/bin/python3

import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		# Init weights and biases
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
		
	def forward(self, inputs):
		# Calc output values from inputs, weights, and biases
		self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


def main():
    nnfs.init()

    X, y = spiral_data(samples=100, classes=3)

    # Create Dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 3)
    

    # Create ReLU activation (to be used with Dense layer)
    activation1 = Activation_ReLU()

    # Make a forward pass of the training data
    dense1.forward(X)  # this will set self.output

    # Forward pass the output to the activation fn
    activation1.forward(dense1.output)

    # Print the first few example
    print(activation1.output[:5])

    # [[0.         0.         0.        ]
    #  [0.         0.00011395 0.        ]
    #  [0.         0.00031729 0.        ]
    #  [0.         0.00052666 0.        ]
    #  [0.         0.00071401 0.        ]]

    # Notice that negative values have been clipped (modified to be zero)


if __name__ == "__main__":
    main()
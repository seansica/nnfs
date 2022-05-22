from re import A
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons))
    
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases 
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs 
        self.output = np.maximum(0, inputs)


# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Suppose we subtract the maximum value from a list of input values. 
        # We would then change the output values to always be in a range from 
        # some negative value up to 0, as the largest number subtracted by itself 
        # returns 0, and any smaller number subtracted by it will result in a 
        # negative number.
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # <-- subtract each input from the 
                                                                             # largest value in the row to mitigate 
                                                                             # "exploding" values
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


def main():
    nnfs.init()

    # Create the data set
    X, y = spiral_data(samples=100, classes=3)
    
    # Create a dense layer w/ 2 input features and 3 output values
    dense1 = Layer_Dense(2, 3)
    
    # Perform ReLU activation on dense1 outputs
    activation1 = Activation_ReLU()

    # Create a 2nd dense layer w/ 3 inputs (to match the previous layers number of outputs) and 3 outputs
    dense2 = Layer_Dense(3,3)

    # Perform softmax activation on dense2 output
    activation2 = Activation_Softmax()

    # Now that everything is initialized, run the data
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Print some examples
    print(activation2.output[:5])

    # [[0.33333334 0.33333334 0.33333334]
    #  [0.3333332  0.3333332  0.33333364]
    #  [0.3333329  0.33333293 0.3333342 ]
    #  [0.3333326  0.33333263 0.33333477]
    #  [0.33333233 0.3333324  0.33333528]]

    # we need a way to calculate how wrong the 
    # neural network is at current predictions and 
    # begin adjusting weights and biases to decrease 
    # error over time. Thus, our next step is to 
    # quantify how wrong the model is through what’s 
    # defined as a loss function.


if __name__ == "__main__":
    main()

import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        # Init weights + biases

        # Note that we’re initializing weights to be (inputs, neurons), rather than 
        # (neurons, inputs). We’re doing this ahead instead of transposing every time we 
        # perform a forward pass, as explained in the previous chapter.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # dimensions = M x n

        # It sometimes may be appropriate to initialize the biases to some non-zero number, 
        # but the most common initialization for biases is 0.
        self.biases = np.zeros((1, n_neurons)) # dimensions = 1 x n

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases
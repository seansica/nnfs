import numpy as np

# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        exp_values = np.exp(inputs)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

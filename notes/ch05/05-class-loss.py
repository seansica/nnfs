import numpy as np

# Common loss class
class Loss:
    # Calculate the data and regularization losses given 
    # model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        return np.mean(sample_losses)
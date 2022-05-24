import numpy as np

LOWER_LIMIT = 1e-7
UPPER_LIMIT = 1 - LOWER_LIMIT


# Common loss class
class Loss:
    # Calculate the data and regularization losses given
    # model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        return np.mean(sample_losses)

    def forward(self, output, y):
        # Will be overridden by sub-class
        pass


# Cross-entropy loss class
class LossCategoricalCrossEntropy(Loss):  # <-- inherits the Loss class
    # Define forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, LOWER_LIMIT, UPPER_LIMIT)

        # Probabilities for target values - only if categorical labels
        dimension = len(y_true.shape)

        if dimension == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
            return -np.log(correct_confidences)
        elif dimension == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            return -np.log(correct_confidences)


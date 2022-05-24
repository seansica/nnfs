import numpy as np

# Since we implemented this to work with sparse labels 
# (as in our training data), we have to add a check if 
# they are one-hot encoded and handle it a bit differently 
# in this new case. 
# 
# The check can be performed by counting 
# the dimensions — if targets are single-dimensional (like 
# a list), they are sparse, but if there are 2 dimensions 
# (like a list of lists), then there is a set of one-hot 
# encoded vectors.

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0]])

dimension = len(class_targets.shape)
i = range(len(softmax_outputs))

if dimension == 1: 
    correct_confidences = softmax_outputs[i, class_targets]
elif dimension == 2: 
    correct_confidences = np.sum( softmax_outputs * class_targets, axis=1 )

loss = -np.log(correct_confidences)
avg_loss = np.mean(loss)

print(avg_loss)
# 0.38506088005216804
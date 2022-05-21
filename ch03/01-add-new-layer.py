import numpy as np

# Add a new layer

# The previous layer has 3 weight sets and 3 biases, so we know it has 3 neurons.

# For the next layer, we can have as many weight sets as we want (because this is 
#   how many neurons this new layer will have), but each of those weight sets must 
#   have 3 discrete weights.

inputs = [[1, 2, 3, 2.5],
            [2., 5., -1., 2],
            [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
  
biases = [2, 3, 0.5]


# The layer that we will create must have its own distinct set of weights and biases.

# The only bit we need to code are the inputs to layer 2 (which are just the outputs 
#   from layer 1)

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]


# Calculate the outputs for input layer 1

layer_1_outputs = np.dot(inputs, np.array(weights).T) + biases

# Recall that layer_1_outputs == layer_2_inputs, so calculate 
#   layer 2 output's exactly like we did before

layer_2_outputs = np.dot(layer_1_outputs, np.array(weights2).T) + biases2

# done

print(layer_2_outputs)

# [[ 0.5031  -1.04185 -2.03875]
#  [ 0.2434  -2.7332  -5.7633 ]
#  [-0.99314  1.41254 -0.35655]]
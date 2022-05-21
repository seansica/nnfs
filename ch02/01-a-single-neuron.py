import numpy as np

# A single neuron with three inputs and one bias
#           ____
# --i1---->|    |
# --i2---->| N  |--out--->
# --i3---->|    |
#           ––––
#

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

outputs = np.dot(weights, inputs) + bias
print(outputs)

# >>> 4.8
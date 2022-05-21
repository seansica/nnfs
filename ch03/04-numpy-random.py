import numpy as np
import nnfs

nnfs.init()

print(np.random.randn(2, 5)) # 2x5 array

# [[ 1.7640524   0.4001572   0.978738    2.2408931   1.867558  ]
#  [-0.9772779   0.95008844 -0.1513572  -0.10321885  0.41059852]]

# In this scenario, we would have 5 input neurons (n_neurons) and two samples

# --

n_inputs = 2
n_neurons = 4

weights = 0.01 * np.random.randn(n_inputs, n_neurons)
biases = np.zeros((1, n_neurons))  # Each neuron must have one bias so we init a 1xN array

print(weights)
print(biases)

# [[ 0.00144044  0.01454273  0.00761038  0.00121675]
#  [ 0.00443863  0.00333674  0.01494079 -0.00205158]]
# [[0. 0. 0. 0.]]
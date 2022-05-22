import math

layer_outputs = [4.8, 1.21, 2.385]

# STEP 1 - exponentiate the outputs using Euler's number, e, which is ~2.71828182846
# y = e ^ x

# e - mathematical constant, we use E here to match a common coding
# style where constants are uppercased
E = math.e

# for each value in a vector, calculate the exponential value
exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output) # ** - power operator in Python

print(exp_values)
# [121.51041751873483, 3.353484652549023, 10.859062664920513]

# To calculate probabilities, we need non-negative numbers. That is why we exponentiate them.


# STEP 2 - once we have exponentiated, convert the numbers to a probability distribution
# i.e., for each value, divide it by the sum of all values

norm_base = sum(exp_values) # sum all values
norm_values = []
for value in exp_values:
    norm_values.append( value / norm_base )

print(norm_values)
# [0.8952826639572619, 0.024708306782099374, 0.0800090292606387]

print('Sum of normalized values:', sum(norm_values))
# Sum of normalized values: 0.9999999999999999


# --
# Now do it the NumPy way
import numpy as np

# Exponentiate each value in the outputs array
exp_values = np.exp(layer_outputs)

# Now normalize each exponentiated value
norm_values = exp_values / np.sum(exp_values)

# To train in batches, we need to convert this functionality to accept layer outputs in batches.
probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# In a 2D array/matrix, axis 0 refers to the rows, and axis 1 refers to the columns.
# This makes so the output has the same size as either the row dimension or the column dimension
#
# e.g.
# np.array([[4.8, 1.21, 2.385],
#           [8.9, -1.81, 0.2],
#           [1.41, 1.051, 0.026]])
#
# axis=0 would mean summing 4.8 + 8.9 + 1.41 and so on
# axis=1 would mean summing 4.8 + 1.21 + 2.385 and so on

layer_outputs = np.array([[4.8, 1.21, 2.385],
                            [8.9, -1.81, 0.2],
                          [1.41, 1.051, 0.026]])

np.sum(layer_outputs, axis=0, keepdims=True)
# array([[15.11 ,      0.451,          2.611]])
#       sum of all     sum of all       sum of all  
#       vals in col 1  vals in col 2    vals in col 3

np.sum(layer_outputs, axis=1, keepdims=True)
# array([[8.395],   <-- sum of all elems in row 1
#        [7.29 ],   <-- sum of all values in row 2
#        [2.487]])  < -- sum of all values in row 3

# If we set keepdims=False, we get a flattened numpy array
# array([8.395, 7.29 , 2.487])

# We want sums of the rows, so we use axis=1. Each sample (each row) 
# represents one probability distribution, i.e. the confidence 
# scoring for one pass-through.
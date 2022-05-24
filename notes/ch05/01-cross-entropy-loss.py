import math

# example output from our output layer
softmax_output = [0.7, 0.1, 0.2]

# one-hot / ground-truth
target_output = [1, 0, 0]

loss = -(
    target_output[0] * math.log(softmax_output[0]) + \
    target_output[1] * math.log(softmax_output[1]) + \
    target_output[2] * math.log(softmax_output[2])
    )

print(loss)
# 0.35667494393873245
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

# ReLU
# y = x if x < 0 else 0

output = []

for i in inputs:
    if i > 0:
        output.append(i)
    else:
        output.append(0)

print(output)
# [0, 2, 0, 3.3, 0, 1.1, 2.2, 0]


# Can we simplifier this code?
output = []
for i in inputs:
    output.append(max(0, i))

print(output)


# NumPy can simplify it even more.
import numpy as np
output = []
output = np.maximum(0, inputs)
print(output)
# [0.  2.  0.  3.3 0.  1.1 2.2 0. ]
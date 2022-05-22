import numpy as np

a = [1,2,3]
b = [2,3,4]

aa = np.array([a])
bb = np.array([b]).T

product = np.dot(aa, bb)

print(product)

# >>>
# [[20]]


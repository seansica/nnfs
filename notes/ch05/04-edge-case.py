import numpy as np

# It is possible for the model to have full confidence for one label,
# thereby making all remaining confidences equal to 0.
#
# Similarly, it is possible that the model will assign full confidence 
# to a value that wasn't the target (i.e., to the wrong target).

# If we try to calcualte the loss of this confidence:
print(-np.log(0))
# RuntimeWarning: divide by zero encountered in log inf

# The negative natural logarithm of 0, in Python with NumPy, equals 
# an infinitely big number, rather than undefined, and prints a 
# warning about a division by 0

print(np.e ** (-np.inf))
# 0.0

y_pred = -np.log(1-1e-7)
y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
print(y_pred_clipped)
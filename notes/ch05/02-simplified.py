import numpy as np

# Consider a scenario with a neural network that performs 
# classification between 3 classes, and classifies in 
# batches of 3.

# Example output:
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])


# Let’s say we’re trying to classify something as a 
# “dog,” “cat,” or “human.” 
#
# A dog is class 0 (at index 0), a cat class 1 (index 1), 
# and a human class 2 (index 2). 
#
# Let’s assume the batch of three sample inputs to this 
# neural network is being mapped to the target values of 
# a dog, cat, and cat. So the targets would be [0,1,1]

class_targets = [0,1,1] # dog, cat, cat

# Print the argmax value from each sample output
for targ_idx, distribution in zip(class_targets, softmax_outputs):
    print(distribution[targ_idx])

# 0.7  <--- index 0, 70% confidence
# 0.5  <--- index 1, 50% confidence
# 0.9  <--- index 1, 90% confidence

# Simplier alternative to printing the argmax values

print(softmax_outputs[
      range(len(softmax_outputs)), class_targets
  ])

# [0.7 0.5 0.9]


# Now apply the negative log to the list to calculate the 
# cross-entropy loss
i = range(len(softmax_outputs))
loss = -np.log( softmax_outputs[i, class_targets] )

print(loss)
# [0.35667494 0.69314718 0.10536052]

# Finally, we want the avg. loss per batch to know how 
# our model is doing during training
avg_loss = np.mean(loss)

print(avg_loss)
# 0.38506088005216804
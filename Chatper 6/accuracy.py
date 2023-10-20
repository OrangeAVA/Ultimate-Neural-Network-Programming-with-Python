import numpy as np


# Probabilities of 3 samples
softmax_outputs = np.array([[0.7, 0.2, 0.1],
                            [0.5, 0.1, 0.4],
                            [0.02, 0.9, 0.08]])


# Target (ground-truth) labels for 3 samples
class_targets = np.array([0, 1, 1])


# Calculate values along the second axis (axis of index 1)
predictions = np.argmax(softmax_outputs, axis=1)


# If targets are one-hot encoded - convert them
if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1)


# True evaluates to 1; False to 0
accuracy = np.mean(predictions == class_targets)
print('acc:', accuracy)


#Output: acc: 0.6666666666666666



# Calculate accuracy from output of activation2 and targets
# Calculate values along the first axis
activation2_output = np.array([[0.1, 0.2, 0.7],
                               [0.9, 0.1, 0.0],
                               [0.3, 0.4, 0.3]])
y = np.array([[1, 0, 0],
              [0, 0, 1],
              [0, 1, 0]])


predictions = np.argmax(activation2_output, axis=1)


if len(y.shape) == 2:
    y = np.argmax(y, axis=1)


accuracy = np.mean(predictions == y)


# Print accuracy
print('acc:', accuracy)


#Output: acc: 0.3333333333333333

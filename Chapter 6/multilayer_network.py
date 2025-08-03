# Processing batch of data through 3 neurons layers
import numpy as np


inputs = [[1, 2, 4, 2.5], [1, 0, 0, 1], [1, 3, 8, 6]]
weights = [[0.1, 0.8, -0.5, 1.1], [0.5, -0.1, 0.6, -0.5], [-0.46, -0.27, 0.17, 0.8]]
biases = [1, 2, 0.5]


weights2 = [[0.11, 0.18, -0.15], [0.5, -0.7, 0.9], [-0.4, -0.7, 0.19]]
biases2 = [-1, 2, 0.2]


# Layer 1 output
layer1_out = np.dot(inputs, np.array(weights).T) + biases


# Layer 2 output
layer2_out = np.dot(layer1_out, np.array(weights2).T) + biases2


print(layer2_out)


#Output: [[-0.3265  3.272  -3.1808]
#         [-0.524   2.456  -1.9204]
#         [-0.4175  7.101  -4.0159]]

# A single neuron using NumPy
import numpy as np


inputs = [1.0, 2.0, 3.0]
weights = [0.2, 0.8, -0.5]
bias = 1.0
outputs = np.dot(weights, inputs) + bias
print(outputs)


# Output: 1.3


# A single Neuron layer using NumPy
import numpy as np


inputs = [1, 2, 4, 2.5]
weights = [[0.1, 0.8, -0.5, 1.1], [0.5, -0.1, 0.6, -0.5], [-0.46, -0.27, 0.17, 0.8]]
biases = [1, 2, 0.5]
layer_out = np.dot(weights, inputs) + biases
print(layer_out)


#Output: [3.45 3.45 2.18]


# Processing batch of data through 3 neurons layers
import numpy as np


inputs = [[1, 2, 4, 2.5], [1, 0, 0, 1], [1, 3, 8, 6]]
weights = [[0.1, 0.8, -0.5, 1.1], [0.5, -0.1, 0.6, -0.5], [-0.46, -0.27, 0.17, 0.8]]
biases = [1, 2, 0.5]
layer_out = np.dot(inputs, np.array(weights).T) + biases
print(layer_out)


#Output: [[3.45 3.45 2.18]
#         [2.2  2.   0.84]
#         [6.1  4.   5.39]]

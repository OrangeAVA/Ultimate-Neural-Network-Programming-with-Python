# Passing input through dense layer followed by applying activation functions

import numpy as np
from sklearn.datasets import make_classification


# ReLU activation function
class Activation_ReLU:
    def forward(self, inputs):
        # Apply ReLU function: replace negative values with zero
        self.output = np.maximum(0, inputs)
        return self.output


# Softmax activation function
class Activation_Softmax:
    def forward(self, inputs):
        # Apply Softmax function to normalize the inputs into probabilities
        # Subtracting the maximum value for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output


# Dense layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with random values and biases with zeros
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        # Perform a forward pass through the layer by multiplying the inputs with weights,
        # adding biases, and storing the result in the 'output' attribute.
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output


# Generate dataset
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.8,
    random_state=42
)


# Create Dense layer with 2 input features and 3 output neurons
dense1 = Layer_Dense(2, 3)


# Apply ReLU activation to the output of the first dense layer
relu_activation = Activation_ReLU()
dense1_output_relu = relu_activation.forward(dense1.forward(X))


# Apply Softmax activation to the output of the ReLU activation
softmax_activation = Activation_Softmax()
dense1_output_softmax = softmax_activation.forward(dense1_output_relu)


# Print the output of the first five samples
print(dense1_output_softmax[:5])


# Output: [[0.32989978 0.34020044 0.32989978]
#          [0.33973738 0.33009912 0.3301635 ]
#          [0.33761584 0.33119208 0.33119208]
#          [0.32925581 0.34148837 0.32925581]
#          [0.33142489 0.33715022 0.33142489]]

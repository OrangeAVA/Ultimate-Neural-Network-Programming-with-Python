## Dense layer and passing data through the Dese layer

import numpy as np
from sklearn.datasets import make_classification


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        """
        Initialize the Layer_Dense class with given number of input and output neurons.
        Initialize weights with random values and biases with zeros.
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        """
        Perform a forward pass through the layer by multiplying the inputs with weights,
        adding biases, and storing the result in the 'output' attribute.
        """
        self.output = np.dot(inputs, self.weights) + self.biases


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


# Perform a forward pass of our training data through this layer
dense1.forward(X)


# Print the output of the first five samples
print(dense1.output[:5])


# Output: [[ 0.00552081 -0.00414039 -0.01683566]
#          [ 0.01143134  0.00287461  0.00670816]
#          [ 0.02203819  0.00209274  0.00040819]
#          [ 0.0523194  -0.00180273 -0.02361715]
#          [ 0.00929377 -0.00188396 -0.00987335]]

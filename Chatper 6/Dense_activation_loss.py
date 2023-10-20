# Passing input through dense layer followed by applying activation functions and finally calculating the loss values

import numpy as np
from sklearn.datasets import make_classification


# Cross-entropy loss
class Loss_CategoricalCrossentropy:
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# ReLU activation function
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output


# Softmax activation function
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output


# Dense layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
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


# Create loss instance
loss = Loss_CategoricalCrossentropy()


# Calculate the loss
loss_value = np.mean(loss.forward(dense1_output_softmax, y))


# Print the loss value
print("Loss:", loss_value)


#Output: Loss: 1.0992215725349792

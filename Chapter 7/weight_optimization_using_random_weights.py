# Weight optimization using randomly generated wieghts

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
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, class_sep=0.8, random_state=42)


# Model building
dense1 = Layer_Dense(2, 3)  
relu_activation = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
softmax_activation = Activation_Softmax()


# Create loss instance
loss = Loss_CategoricalCrossentropy()


lowest_loss = 999


best_dense1_wt = dense1.weights.copy()
best_dense1_b = dense1.biases.copy()
best_dense2_wt = dense2.weights.copy()
best_dense2_b = dense2.biases.copy()


for iteration in range(10000):
    dense1.weights += 0.05*np.random.randn(2,3)
    dense1.biases += 0.05*np.random.randn(1,3)
    dense2.weights += 0.05*np.random.randn(3,3)
    dense2.weights += 0.05*np.random.randn(1,3)


    dense1.forward(X)
    relu_activation.forward(dense1.output)
    dense2.forward(relu_activation.output)
    softmax_activation.forward(dense2.output)


    loss_iteration = np.mean(loss.forward(softmax_activation.output, y))


    predictions = np.argmax(softmax_activation.output, axis=1)
    accuracy = np.mean(predictions == y)


    if loss_iteration < lowest_loss:
        print('New set of weights found, iteration:', iteration,
        'loss:', loss_iteration, 'acc:', accuracy)
        best_dense1_wt = dense1.weights.copy()
        best_dense1_b = dense1.biases.copy()
        best_dense2_wt = dense2.weights.copy()
        best_dense2_b = dense2.biases.copy()
        lowest_loss = loss_iteration
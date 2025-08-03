import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


# Dense layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with random values scaled by 0.01
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # Initialize biases as zeros
        self.biases = np.zeros((1, n_neurons))
        # Initialize gradient variables as None
        self.dweights = None
        self.dbiases = None


    def forward(self, inputs):
        # Save the inputs for later use in backward propagation
        self.inputs = inputs
        # Perform the dot product of inputs and weights, and add biases
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output


    def backward(self, dvalues):
        # Calculate the gradients of weights, biases, and inputs
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


# ReLU activation function
class Activation_ReLU:
    def forward(self, inputs):
        # Save the inputs for later use in backward propagation
        self.inputs = inputs
        # Apply the ReLU activation function element-wise
        self.output = np.maximum(0, inputs)
        return self.output


    def backward(self, dvalues):
        # Create a copy of the gradients of the outputs
        self.dinputs = dvalues.copy()
        # Set gradients of the inputs corresponding to the negative input values to 0
        self.dinputs[self.inputs <= 0] = 0


# Softmax activation function
class Activation_Softmax:
    def forward(self, inputs):
        # Calculate exponential values of inputs while normalizing them
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Calculate probabilities by dividing each exponential value by the sum of all exponential values
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        # Save the output for later use in backward propagation
        self.output = probabilities
        return self.output


    def backward(self, dvalues):
        # Create an array to store the gradients of the inputs
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Reshape the single_output array into a column vector
            single_output = single_output.reshape(-1, 1)
            # Calculate the Jacobian matrix for the softmax function
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate the gradient of the inputs using the Jacobian matrix and the gradients of the outputs
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


# Cross-entropy loss
class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            # For sparse label format, extract the correct confidence scores
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            # For one-hot encoded label format, calculate the dot product of probabilities and labels
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        # Calculate negative log-likelihood loss
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            # For sparse label format, convert y_true to one-hot encoded format
            y_true = np.eye(labels)[y_true]
        
        # Calculate gradient of the loss with respect to the inputs
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:
    # Initialize optimizer – set settings,
    # learning rate of 1. Is default for this optimizer
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
        # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases


# Generate dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, class_sep=0.8, random_state=42)


# Create layers
dense1 = Layer_Dense(2, 3)
relu_activation = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
softmax_activation = Activation_Softmax()


# Create loss function
loss_function = Loss_CategoricalCrossentropy()


# Create optimizer
optimizer = Optimizer_SGD(learning_rate=0.1)


# Training loop
losses = []
accuracies = []
for epoch in range(500):


    # Forward pass
    dense1_output = dense1.forward(X)
    relu_activation_output = relu_activation.forward(dense1_output)
    dense2_output = dense2.forward(relu_activation_output)
    softmax_activation_output = softmax_activation.forward(dense2_output)


    # Calculate loss
    loss = np.mean(loss_function.forward(softmax_activation_output, y))
    losses.append(loss)
    predictions = np.argmax(softmax_activation_output, axis=1)
    accuracy = np.mean(predictions == y)
    # Append the accuracy to accuracies list
    accuracies.append(accuracy)
    print(f'Epoch: {epoch}, Accuracy: {accuracy:.3f}, Loss: {loss:.3f}')


    # Backward pass
    loss_function.backward(softmax_activation_output, y)
    softmax_activation.backward(loss_function.dinputs)
    dense2.backward(softmax_activation.dinputs)
    relu_activation.backward(dense2.dinputs)
    dense1.backward(relu_activation.dinputs)


    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)


plt.figure()
plt.xlabel('Epoch')


# Create twin axes that share the same x-axis
ax1 = plt.gca()
ax2 = ax1.twinx()


# Plot loss (on y-axis of ax1)
ax1.plot(losses, color='blue', label='Loss')
ax1.set_ylabel('Loss', color='blue')
ax1.tick_params('y', colors='blue')


# Plot accuracy (on y-axis of ax2)
ax2.plot(accuracies, color='red', label='Accuracy')
ax2.set_ylabel('Accuracy', color='red')
ax2.tick_params('y', colors='red')


plt.title('Loss and Accuracy over epochs')
plt.show()

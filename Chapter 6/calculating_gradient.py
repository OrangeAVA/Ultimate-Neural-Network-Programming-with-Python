import numpy as np
from sklearn.datasets import make_classification


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



# Generate dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                            n_redundant=0, n_clusters_per_class=1, class_sep=0.8, random_state=42)


# Forward pass
dense1 = Layer_Dense(2, 3)
relu_activation = Activation_ReLU()


dense2 = Layer_Dense(3, 3)
softmax_activation = Activation_Softmax()


dense1_output = dense1.forward(X)
relu_activation_output = relu_activation.forward(dense1_output)


dense2_output = dense2.forward(relu_activation_output)
softmax_activation_output = softmax_activation.forward(dense2_output)


# Create loss instance
loss = Loss_CategoricalCrossentropy()


loss_iteration = np.mean(loss.forward(softmax_activation_output, y))
predictions = np.argmax(softmax_activation_output, axis=1)
accuracy = np.mean(predictions == y)
print("Accuracy: ", accuracy)


# Backward pass
loss.backward(softmax_activation_output, y)
softmax_activation.backward(loss.dinputs)
dense2.backward(softmax_activation.dinputs)
relu_activation.backward(dense2.dinputs)
dense1.backward(relu_activation.dinputs)


# Print gradients
print(dense1.dweights, "\n")
print(dense1.dbiases, "\n")
print(dense2.dweights, "\n")
print(dense2.dbiases)


# Accuracy:  0.501
# [[-7.91311439e-04  9.10036274e-05 -3.62927206e-03]
#  [-2.42775242e-03  1.08710225e-03  2.21541699e-03]] 


# [[-0.00209153  0.00073589 -0.00212986]] 


# [[ 0.00141515 -0.00286223  0.00144708]
#  [ 0.00043136 -0.00086976  0.0004384 ]
#  [-0.001922   -0.0027196   0.0046416 ]] 


# [[-0.16761353 -0.16567281  0.33328634]]

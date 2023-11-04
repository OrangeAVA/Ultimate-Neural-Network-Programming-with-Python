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


class Optimizer_Adam:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        # Learning rate of 0.001 is default for this optimizer
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        # Decay rate for the learning rate
        self.decay = decay
        # Number of steps taken so far
        self.iterations = 0
        # Small number to prevent division by zero
        self.epsilon = epsilon
        # Coefficients used for computing running averages of gradient and its square
        self.beta_1 = beta_1
        self.beta_2 = beta_2


    # Call just before the update step, update learning rate if decay is set
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))


    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays, create them
        # filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)


        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases


        # Get corrected momentum
        # self.iteration is 0 at first pass and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))


        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2


        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))


        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
        
    # Call after any parameter updates
    def post_update_params(self):
        self.iterations += 1



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
#optimizer = Optimizer_SGD(learning_rate=0.1)
optimizer = Optimizer_Adam(learning_rate=0.01)


# Training loop
losses = []
accuracies = []
for epoch in range(500):


    # Call pre_update_params at the start of each epoch
    optimizer.pre_update_params()


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


    # Call post_update_params at the end of each epoch
    optimizer.post_update_params()


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
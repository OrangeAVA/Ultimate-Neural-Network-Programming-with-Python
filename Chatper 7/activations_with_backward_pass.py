# ReLU activation function
class Activation_ReLU:
    def forward(self, inputs):
        # Save the inputs for later use in backward propagation
        self.inputs = inputs
        # Apply the ReLU activation function element-wise
        self.output = np.maximum(0, inputs)
        return self.output




    # backward method for ReLU activation function
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


# Added backward method for softmax activation
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
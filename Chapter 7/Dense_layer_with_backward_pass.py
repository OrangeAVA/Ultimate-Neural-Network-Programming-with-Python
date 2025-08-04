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


    	# added backward method for Dense layer
def backward(self, dvalues):
        # Calculate the gradients of weights, biases, and inputs
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
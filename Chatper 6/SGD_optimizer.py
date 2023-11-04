class Optimizer_SGD:
    # Initialize optimizer – set settings,
    # learning rate of 1. Is default for this optimizer
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
        # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

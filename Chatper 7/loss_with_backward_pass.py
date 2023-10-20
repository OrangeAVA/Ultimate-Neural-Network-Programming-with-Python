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




# Added backward method for categorical cross-entropy
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            # For sparse label format, convert y_true to one-hot encoded format
            y_true = np.eye(labels)[y_true]
        
        # Calculate gradient of the loss with respect to the inputs
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
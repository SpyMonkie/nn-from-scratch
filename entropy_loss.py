import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

# Common Loss class
class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate the sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss

# Loss function for categorical crossentropy
# This loss function is used for multi-class classification problems
class Loss_CategoricalCrossentropy(Loss):
    # forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by zero
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values -
        # only if categroical labels
        if len(y_true.shape) == 1:
            # if inputs are not one-hot encoded use array indexing for getting the highest probability for each batch
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            # if inputs are one-hot encoded, multiply predictions with true values
            # and sum them up to get the correct confidences
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)


        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, convert them to one-hot encoded
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

softmax_outputs = np.array([[0.7, 0.1, 0.2],
[0.1, 0.5, 0.4],
[0.02, 0.9, 0.08]])
class_targets = np.array([[1, 0, 0],
[0, 1, 0],
[0, 1, 0]])
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(softmax_outputs, class_targets)
print(loss)

# def main():
#     softmax_outputs = np.array([[0.7, 0.1, 0.2],
#     [0.1, 0.5, 0.4],
#     [0.02, 0.9, 0.08]])
#     class_targets = np.array([[1, 0, 0],
#     [0, 1, 0],
#     [0, 1, 0]])
#     loss_function = Loss_CategoricalCrossentropy()
#     loss = loss_function.calculate(softmax_outputs, class_targets)
#     print(loss)

# if __name__ == "__main__":
#     main()
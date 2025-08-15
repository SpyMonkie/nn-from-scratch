import numpy as np

class Layer_Dropout:
    def __init__(self, rate):
        # Store the dropout rate, invert it to get the success rate
        # For example, for a dropout rate of 0.1, we need a success rate of 0.9
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs):
        # Save the input values
        self.inputs = inputs
        # Generate and save the scaled binary mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply the mask to the inputs
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Apply the mask to the gradients
        self.dinputs = dvalues * self.binary_mask
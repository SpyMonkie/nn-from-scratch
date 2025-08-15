import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data
import matplotlib.pyplot as plt
from dense_layers import Layer_Dense, Activation_ReLU, Activation_Softmax, Activation_Softmax_Loss_CategoricalCrossentropy
from entropy_loss import Loss_CategoricalCrossentropy, Loss
from SGD_optimizer import Optimizer_SGD

nnfs.init()

# optimization based on iteration of previous weights and biases
# until loss is below a certain threshold
# fails on complex data, but works on simple data like vertical_data

def main():
    # Create spiral data
    # two inputs, X1 and X2, and three classes
    # X, y = spiral_data(samples=100, classes=3)
    X, y = spiral_data(samples=100, classes=3)

    plt.figure(figsize=(8, 6))
    plt.title('Data Input')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg', s=40, edgecolors='k')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    # Create Dense Layer with 2 inputs features and 64 neurons (64 output values)
    dense1 = Layer_Dense(2, 64)

    # Create ReLU activation function (to be use with the dense layer)
    activation1 = Activation_ReLU()

    # Create second Dense Layer with 64 input features (as we take output
    # of previous layer here) and 3 output values
    dense2 = Layer_Dense(64, 3)

    # Create Softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    # Create SGD optimizer
    optimizer = Optimizer_SGD()


    # Helper variables
    # lowest_loss = float('inf')  # Initialize lowest loss
    lowest_loss = 9999999
    best_dense1_weights = dense1.weights.copy() # Copy initial weights
    best_dense1_biases = dense1.biases.copy() # Copy initial biases
    best_dense2_weights = dense2.weights.copy() # Copy initial weights
    best_dense2_biases = dense2.biases.copy() # Copy initial biases


    for iteration in range(100000):
        # Perform a forward pass of our training data through this layer
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss_activation.forward(dense2.output, y)

        # Calculate loss
        loss = loss_activation.loss.calculate(loss_activation.output, y)
        # print(f'Loss: {loss}')

        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y_labels = np.argmax(y, axis=1)
        else:
            y_labels = y
        accuracy = np.mean(predictions == y_labels)

        if not iteration % 100:
            print(f'Iteration {iteration}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases using SGD optimizer
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)

    # Print final results
    print(f'Final Loss: {lowest_loss:.4f}, Best Weights and Biases Found')
    print('Best Dense1 Weights:\n', best_dense1_weights)
    print('Best Dense1 Biases:\n', best_dense1_biases)
    print('Best Dense2 Weights:\n', best_dense2_weights)
    print('Best Dense2 Biases:\n', best_dense2_biases)

    # Test different points to see how the model performs
    test_points = np.array([[0.5, 0.5], [-0.5, -0.5], [1.0, 1.0], [-1.0, -1.0]])
    for point in test_points:
        dense1.forward(point)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        output = loss_activation.activation.output
        predicted_class = np.argmax(output)
        print(f'Input: {point}, Predicted Class: {predicted_class}, Output Probabilities: {output}')

if __name__ == "__main__":
    main()
import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data
import matplotlib.pyplot as plt
from dense_layers import Layer_Dense, Activation_ReLU, Activation_Softmax
from entropy_loss import Loss_CategoricalCrossentropy, Loss

nnfs.init()

# optimization based on iteration of previous weights and biases
# until loss is below a certain threshold
# fails on complex data, but works on simple data like vertical_data

def main():
    # Create spiral data
    # two inputs, X1 and X2, and three classes
    # X, y = spiral_data(samples=100, classes=3)
    X, y = vertical_data(samples=100, classes=3)

    plt.figure(figsize=(8, 6))
    plt.title('Data Input')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg', s=40, edgecolors='k')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    # Create Dense Layer with 2 inputs features and 3 neurons (3 output values)
    dense1 = Layer_Dense(2, 3)

    # Create ReLU activation function (to be use with the dense layer)
    activation1 = Activation_ReLU()

    # Create second Dense Lyaer with 3 input features (as we take output
    # of previous layer here) and 3 output values
    dense2 = Layer_Dense(3, 3)

    # Create Softmax activation function (to be use with the dense layer)
    activation2 = Activation_Softmax()

    # Create loss function
    loss_function = Loss_CategoricalCrossentropy()

    # Helper variables
    # lowest_loss = float('inf')  # Initialize lowest loss
    lowest_loss = 9999999
    best_dense1_weights = dense1.weights.copy() # Copy initial weights
    best_dense1_biases = dense1.biases.copy() # Copy initial biases
    best_dense2_weights = dense2.weights.copy() # Copy initial weights
    best_dense2_biases = dense2.biases.copy() # Copy initial biases

    # for iteration in range(100000):
    iteration = 0
    while lowest_loss > 0.1:
        iteration += 1
        dense1.weights += 0.05 * np.random.randn(2, 3)  # Randomize weights
        dense1.biases += 0.05 * np.random.randn(1, 3)  # Randomize biases
        dense2.weights += 0.05 * np.random.randn(3, 3)  # Randomize weights
        dense2.biases += 0.05 * np.random.randn(1, 3)  # Randomize biases
        # Perform a forward pass of our training data through this layer
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        # Calculate the loss
        loss = loss_function.calculate(activation2.output, y)
        # print(f'Loss: {loss}')

        predictions = np.argmax(activation2.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        accuracy = np.mean(predictions == y)
        # print(f'Accuracy: {accuracy * 100:.2f}%')

        # print(activation2.output[:5])  # Print the first 5 output probabilities

        # # Plot the output of the forward pass
        # plt.figure(figsize=(8, 6))
        # plt.title('Output of the Forward Pass')
        # plt.scatter(activation2.output[:, 0], activation2.output[:, 1], c
        # =y, cmap='brg', s=40, edgecolors='k')
        # plt.xlabel('Output 1')
        # plt.ylabel('Output 2')
        # plt.show()

        # if loss is smaller - print and save weights and biases aside
        if loss < lowest_loss:
            print("New set of weights found, iteration: ", iteration, " loss:", loss, " accuracy:", accuracy * 100)
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss

            #  # only show figure if new loss/weights are found
            #  # Plot the output of the forward pass
            # plt.figure(figsize=(8, 6))
            # plt.title('Output of the Forward Pass')
            # plt.scatter(activation2.output[:, 0], activation2.output[:, 1], c
            # =y, cmap='brg', s=40, edgecolors='k')
            # plt.xlabel('Output 1')
            # plt.ylabel('Output 2')
            # plt.show()
        # else - set weights and biases to the best found so far
        else:
            dense1.weights = best_dense1_weights
            dense1.biases = best_dense1_biases
            dense2.weights = best_dense2_weights
            dense2.biases = best_dense2_biases




if __name__ == "__main__":
    main()
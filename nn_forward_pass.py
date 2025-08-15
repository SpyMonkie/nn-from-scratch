import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data
import matplotlib.pyplot as plt
from dense_layers import Layer_Dense, Activation_ReLU, Activation_Softmax
from entropy_loss import Loss_CategoricalCrossentropy, Loss

nnfs.init()

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

    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Make a forward pass through the activation function
    # it takes the output of the first dense layer here
    activation1.forward(dense1.output)
    # Perform a forward pass of the output of the first activation function
    dense2.forward(activation1.output)
    # Make a forward pass through the
    activation2.forward(dense2.output)

    # Calculate the loss
    loss = loss_function.calculate(activation2.output, y)
    print(f'Loss: {loss}')

    predictions = np.argmax(activation2.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # print(activation2.output[:5])  # Print the first 5 output probabilities

    # Plot the output of the forward pass
    plt.figure(figsize=(8, 6))
    plt.title('Output of the Forward Pass')
    plt.scatter(activation2.output[:, 0], activation2.output[:, 1], c
    =y, cmap='brg', s=40, edgecolors='k')
    plt.xlabel('Output 1')
    plt.ylabel('Output 2')
    plt.show()


    # Plot the output of the dense layer
    # plt.scatter(activation2.output[:, 0], activation2.output[:, 1], c=y, cmap='brg', s=40, edgecolors='k')
    # Plot the output of the dense layers
    # plt.show()


if __name__ == "__main__":
    main()
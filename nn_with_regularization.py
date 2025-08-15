import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data
import matplotlib.pyplot as plt
from dense_layers_regularization import Layer_Dense, Activation_ReLU, Activation_Softmax, Activation_Softmax_Loss_CategoricalCrossentropy
from entropy_loss_regularization import Loss_CategoricalCrossentropy, Loss
from SGD_optimizer import Optimizer_SGD
from ADAGRAD_optimizer import Optimizer_Adagrad
from RMSprop_optimizer import Optimizer_RMSprop
from ADAM_optimizer import Optimizer_Adam

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
    dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

    # Create ReLU activation function (to be use with the dense layer)
    activation1 = Activation_ReLU()

    # Create second Dense Layer with 64 input features (as we take output
    # of previous layer here) and 3 output values
    dense2 = Layer_Dense(64, 3)

    # Create Softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    # Create SGD optimizer
    # optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)

    # Create ADAGRAD optimizer
    # optimizer = Optimizer_Adagrad(decay=1e-4)

    # Create RMSprop optimizer
    # optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)

    # Create ADAM optimizer
    optimizer = Optimizer_Adam(learning_rate=0.02, decay=1e-5)

    for iteration in range(10001):
        # Perform a forward pass of our training data through this layer
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        data_loss = loss_activation.forward(dense2.output, y)
        # print(f'Loss: {loss}')

        regularization_loss = (
            loss_activation.loss.regularization_loss(dense1) +
            loss_activation.loss.regularization_loss(dense2)
        )

        # Calculate overall loss
        loss = data_loss + regularization_loss

        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y_labels = np.argmax(y, axis=1)
        else:
            y_labels = y
        accuracy = np.mean(predictions == y_labels)

        if not iteration % 100:
            print(f'Iteration {iteration}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f} lr: {optimizer.current_learning_rate:.4f}')

        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases using SGD optimizer
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

    # Print final results
    # print(f'Final Loss: {lowest_loss:.4f}, Best Weights and Biases Found')
    # print('Best Dense1 Weights:\n', best_dense1_weights)
    # print('Best Dense1 Biases:\n', best_dense1_biases)
    # print('Best Dense2 Weights:\n', best_dense2_weights)
    # print('Best Dense2 Biases:\n', best_dense2_biases)

if __name__ == "__main__":
    main()
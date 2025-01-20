import numpy as np


class NeuralNetwork:
    def __init__(self, layers):
        """
        layers: list of integers representing the number of neurons in each layer
        """
        self.layers = layers
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for i in range(len(layers) - 1):
            # use He Normal initialization - paper "Delving Deep into Rectifiers"
            w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self.relu(z)
            self.activations.append(activation)

        # Output layer
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = self.softmax(z)
        self.activations.append(output)

        return output

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        delta = self.activations[-1]
        delta[range(m), y] -= 1
        delta /= m

        gradients_w = []
        gradients_b = []

        for i in range(len(self.weights) - 1, -1, -1):
            gradients_w.insert(0, np.dot(self.activations[i].T, delta))
            gradients_b.insert(0, np.sum(delta, axis=0, keepdims=True))

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(
                    self.z_values[i - 1]
                )

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.biases[i] -= learning_rate * gradients_b[i]

    def train(self, X, y, epochs=10, batch_size=32, learning_rate=0.01):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            # Shuffle the data
            shuffle_idx = np.random.permutation(n_samples)
            X_shuffled = X[shuffle_idx]
            y_shuffled = y[shuffle_idx]

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # Forward and backward pass
                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)

            # Calculate accuracy for this epoch
            predictions = np.argmax(self.forward(X), axis=1)
            accuracy = np.mean(predictions == y)
            print(f"Epoch {epoch+1}/{epochs}, Accuracy: {accuracy:.4f}")

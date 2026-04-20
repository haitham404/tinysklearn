import numpy as np
from .NeuronLayer import NeuronLayer


class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs, learning_rate=0.1):
        """
        Initialize a 2-layer neural network (1 hidden + 1 output).

        Args:
            num_inputs: Number of input features
            num_hidden: Number of hidden layer neurons
            num_outputs: Number of output neurons
            learning_rate: Learning rate for weight updates
        """
        self.learning_rate = learning_rate
        self.hidden_layer = NeuronLayer(num_hidden, num_inputs)
        self.output_layer = NeuronLayer(num_outputs, num_hidden)

    def feedforward(self, X):
        """
        Forward pass through the network.

        Args:
            X: Input data

        Returns:
            Network output predictions
        """
        hidden_output = self.hidden_layer.feedforward(X)
        final_output = self.output_layer.feedforward(hidden_output)
        return final_output

    def train_step(self, X, y):
        """
        Single training step: forward pass, backpropagation, weight update.

        Args:
            X: Input data
            y: Target values

        Returns:
            Network predictions for this batch
        """
        # Ensure proper shape
        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Forward pass
        output = self.feedforward(X)

        # Backward pass - Output layer delta
        self.output_layer.compute_delta(target=y)

        # Backward pass - Hidden layer delta
        weights_to_next = np.array([n.weights for n in self.output_layer.neurons])
        self.hidden_layer.compute_delta(
            next_layer_weights=weights_to_next,
            next_layer_delta=self.output_layer.deltas
        )

        # Update weights
        self.output_layer.update_weights(self.hidden_layer.outputs, self.learning_rate)
        self.hidden_layer.update_weights(X, self.learning_rate)

        return output

    def fit(self, X, y, epochs=100, batch_size=32, verbose=True):
        """
        Train the network for multiple epochs.

        Args:
            X: Training data
            y: Target values
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print training progress

        Returns:
            self (for method chaining)
        """
        X = np.array(X)
        y = np.array(y)

        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0

            # Mini-batch training
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                output = self.train_step(X_batch, y_batch)

                # Compute MSE loss
                loss = np.mean((output - y_batch.reshape(-1, 1)) ** 2)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        return self

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Input data

        Returns:
            Predictions
        """
        return self.feedforward(X)
import numpy as np
from .Neuron import Neuron


class NeuronLayer:
    def __init__(self, num_neurons, num_inputs):
        """
        Initialize a layer of neurons.

        Args:
            num_neurons: Number of neurons in this layer
            num_inputs: Number of inputs per neuron
        """
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
        self.outputs = None
        self.deltas = None

    def feedforward(self, X):
        """
        Forward pass through all neurons in the layer.

        Args:
            X: Input data (can be 1D or 2D)

        Returns:
            Layer outputs (1D if single sample, 2D if batch)
        """
        # Handle both single sample and batch
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Compute output for each sample
        outputs = []
        for sample in X:
            layer_output = [neuron.compute_output(sample) for neuron in self.neurons]
            outputs.append(layer_output)

        self.outputs = np.array(outputs)
        return self.outputs

    def compute_delta(self, target=None, next_layer_weights=None, next_layer_delta=None):
        """
        Compute error deltas for backpropagation.

        For output layer: delta = (output - target) * σ'(output)
        For hidden layer: delta = (weights^T · next_delta) * σ'(output)

        Args:
            target: Target values (for output layer)
            next_layer_weights: Weights from next layer (for hidden layer)
            next_layer_delta: Deltas from next layer (for hidden layer)
        """
        if target is not None:
            # Output layer: compare with target
            self.deltas = (self.outputs - target) * self._sigmoid_derivative(self.outputs)
        else:
            # Hidden layer: backpropagate from next layer
            next_layer_weights = np.array(next_layer_weights)
            deltas = []
            for i, neuron in enumerate(self.neurons):
                # Get the weight from this neuron to all neurons in next layer
                incoming_weights = next_layer_weights[:, i]
                weighted_sum = np.dot(incoming_weights, next_layer_delta)
                delta = weighted_sum * self._sigmoid_derivative(neuron.output)
                deltas.append(delta)
            self.deltas = np.array(deltas)

    def update_weights(self, layer_input, learning_rate):
        """
        Update neuron weights using gradient descent.

        w_new = w_old - lr * delta * input

        Args:
            layer_input: Input to this layer
            learning_rate: Learning rate for weight updates
        """
        if layer_input.ndim == 1:
            layer_input = layer_input.reshape(1, -1)

        for i, neuron in enumerate(self.neurons):
            # Accumulate gradients across batch
            gradient = np.mean(self.deltas[:, i:i + 1] * layer_input, axis=0)
            neuron.weights -= learning_rate * gradient
            neuron.bias -= learning_rate * np.mean(self.deltas[:, i])

    @staticmethod
    def _sigmoid_derivative(outputs):
        """Derivative of sigmoid"""
        return outputs * (1 - outputs)
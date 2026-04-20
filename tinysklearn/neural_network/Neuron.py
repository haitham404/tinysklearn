import numpy as np


class Neuron:
    def __init__(self, num_inputs):
        """
        Initialize a neuron with random weights and zero bias.

        Args:
            num_inputs: Number of input connections
        """
        self.weights = np.random.randn(num_inputs) * 0.01
        self.bias = 0.0
        self.output = None
        self.input = None
        self.delta = None

    def compute_output(self, inputs):
        """
        Compute neuron output: z = w·x + b
        Applies sigmoid activation for hidden/output layers.

        Args:
            inputs: Input array (1D numpy array)

        Returns:
            Activated output
        """
        self.input = inputs
        z = np.dot(self.weights, inputs) + self.bias
        self.output = self._sigmoid(z)
        return self.output

    @staticmethod
    def _sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def _sigmoid_derivative(x):
        """Derivative of sigmoid: σ'(x) = σ(x)(1-σ(x))"""
        return x * (1 - x)
"""Dense layers and a small multilayer perceptron with explicit backpropagation."""

import numpy as np

from .activations import relu, relu_derivative


class Dense:
    """Affine layer ``x @ weight + bias`` for batches of row vectors."""

    def __init__(self, input_size: int, output_size: int, rng: np.random.Generator):
        if input_size < 1 or output_size < 1:
            raise ValueError("layer dimensions must be positive")
        self.weight = rng.standard_normal((input_size, output_size)) * np.sqrt(2 / input_size)
        self.bias = np.zeros(output_size)
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        self._input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.weight.shape[0]:
            raise ValueError("input shape does not match layer")
        self._input = x
        return x @ self.weight + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backpropagate upstream gradient and store parameter gradients.

        Gradients are summed across the batch. The loss function is responsible
        for any batch averaging, avoiding accidental double scaling.
        """
        if self._input is None:
            raise RuntimeError("call forward before backward")
        grad_output = np.asarray(grad_output, dtype=float)
        if grad_output.shape != (self._input.shape[0], self.weight.shape[1]):
            raise ValueError("upstream gradient has the wrong shape")
        self.grad_weight = self._input.T @ grad_output
        self.grad_bias = grad_output.sum(axis=0)
        return grad_output @ self.weight.T

    def parameters_and_gradients(self):
        """Yield mutable parameters with their latest gradients."""
        yield self.weight, self.grad_weight
        yield self.bias, self.grad_bias


class MLP:
    """Dense network with ReLU hidden layers and raw output logits."""

    def __init__(self, sizes: list[int], seed: int = 0):
        if len(sizes) < 2:
            raise ValueError("sizes must include input and output dimensions")
        rng = np.random.default_rng(seed)
        self.layers = [Dense(a, b, rng) for a, b in zip(sizes[:-1], sizes[1:])]
        self._hidden_pre_activations = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._hidden_pre_activations = []
        for layer in self.layers[:-1]:
            x = layer.forward(x)
            self._hidden_pre_activations.append(x)
            x = relu(x)
        return self.layers[-1].forward(x)

    def backward(self, grad_logits: np.ndarray) -> np.ndarray:
        if len(self._hidden_pre_activations) != len(self.layers) - 1:
            raise RuntimeError("call forward before backward")
        grad = self.layers[-1].backward(grad_logits)
        for layer, pre_activation in zip(
            reversed(self.layers[:-1]), reversed(self._hidden_pre_activations)
        ):
            grad = layer.backward(grad * relu_derivative(pre_activation))
        return grad

    def parameters_and_gradients(self):
        for layer in self.layers:
            yield from layer.parameters_and_gradients()

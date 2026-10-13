"""Recurrent neural network layers implemented with explicit NumPy backpropagation."""

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    positive = x >= 0
    result = np.empty_like(x, dtype=float)
    result[positive] = 1 / (1 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1 + exp_x)
    return result


class LSTM:
    """Batch-first LSTM layer with one output vector per time step."""

    def __init__(self, input_size: int, hidden_size: int, seed: int = 0):
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed)
        scale = np.sqrt(1 / (input_size + hidden_size))
        self.weight = rng.uniform(-scale, scale, (input_size + hidden_size, 4 * hidden_size))
        self.bias = np.zeros(4 * hidden_size)
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        self._cache = None
        self._input_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run a sequence with zero initial hidden and cell states."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 3 or x.shape[2] != self.input_size:
            raise ValueError("x must have shape (batch, time, input_size)")
        batch, steps, _ = x.shape
        hidden = np.zeros((batch, self.hidden_size))
        cell = np.zeros_like(hidden)
        outputs = np.empty((batch, steps, self.hidden_size))
        cache = []
        for step in range(steps):
            inputs = np.concatenate((x[:, step], hidden), axis=1)
            gates = inputs @ self.weight + self.bias
            input_gate, forget_gate, candidate, output_gate = np.split(gates, 4, axis=1)
            input_gate, forget_gate, output_gate = map(_sigmoid, (input_gate, forget_gate, output_gate))
            candidate = np.tanh(candidate)
            previous_cell = cell
            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * np.tanh(cell)
            outputs[:, step] = hidden
            cache.append((inputs, input_gate, forget_gate, candidate, output_gate, previous_cell, cell))
        self._cache = cache
        self._input_shape = x.shape
        return outputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backpropagate through time and return the gradient with respect to x."""
        if self._cache is None or self._input_shape is None:
            raise RuntimeError("call forward before backward")
        grad_output = np.asarray(grad_output, dtype=float)
        expected = (self._input_shape[0], self._input_shape[1], self.hidden_size)
        if grad_output.shape != expected:
            raise ValueError("grad_output must match the forward output shape")
        batch, steps, _ = self._input_shape
        grad_x = np.zeros(self._input_shape)
        grad_hidden = np.zeros((batch, self.hidden_size))
        grad_cell = np.zeros_like(grad_hidden)
        self.grad_weight.fill(0)
        self.grad_bias.fill(0)
        for step in range(steps - 1, -1, -1):
            inputs, input_gate, forget_gate, candidate, output_gate, previous_cell, cell = self._cache[step]
            grad_hidden += grad_output[:, step]
            tanh_cell = np.tanh(cell)
            grad_output_gate = grad_hidden * tanh_cell
            grad_cell_total = grad_cell + grad_hidden * output_gate * (1 - tanh_cell**2)
            grad_forget = grad_cell_total * previous_cell
            grad_input_gate = grad_cell_total * candidate
            grad_candidate = grad_cell_total * input_gate
            grad_cell = grad_cell_total * forget_gate
            gate_gradients = np.concatenate((
                grad_input_gate * input_gate * (1 - input_gate),
                grad_forget * forget_gate * (1 - forget_gate),
                grad_candidate * (1 - candidate**2),
                grad_output_gate * output_gate * (1 - output_gate),
            ), axis=1)
            self.grad_weight += inputs.T @ gate_gradients
            self.grad_bias += gate_gradients.sum(axis=0)
            grad_inputs = gate_gradients @ self.weight.T
            grad_x[:, step] = grad_inputs[:, : self.input_size]
            grad_hidden = grad_inputs[:, self.input_size:]
        return grad_x

    def parameters_and_gradients(self):
        """Yield trainable parameters with their latest gradients."""
        yield self.weight, self.grad_weight
        yield self.bias, self.grad_bias

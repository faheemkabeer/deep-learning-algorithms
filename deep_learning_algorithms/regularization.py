"""Dropout regularization for NumPy neural networks."""

import numpy as np


class Dropout:
    """Randomly suppress activations during training using inverted scaling."""

    def __init__(self, probability: float = 0.5, seed: int = 0):
        if not 0 <= probability < 1:
            raise ValueError("dropout probability must be in [0, 1)")
        self.probability = probability
        self._rng = np.random.default_rng(seed)
        self._mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if training and self.probability:
            self._mask = (
                self._rng.random(x.shape) >= self.probability
            ) / (1 - self.probability)
        else:
            self._mask = np.ones_like(x)
        return x * self._mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("call forward before backward")
        grad_output = np.asarray(grad_output, dtype=float)
        if grad_output.shape != self._mask.shape:
            raise ValueError("grad_output shape does not match forward input")
        return grad_output * self._mask

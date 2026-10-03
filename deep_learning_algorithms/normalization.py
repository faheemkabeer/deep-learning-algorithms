"""Batch normalization for two-dimensional (batch, features) arrays."""

import numpy as np


class BatchNorm1d:
    """Normalize each feature and learn a scale and offset.

    Training uses the current batch and updates running statistics. Evaluation
    uses the running statistics collected during training.
    """

    def __init__(self, features: int, momentum: float = 0.1, epsilon: float = 1e-5):
        if features < 1 or not 0 < momentum <= 1 or epsilon <= 0:
            raise ValueError("invalid BatchNorm1d settings")
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
        self.grad_gamma = np.zeros(features)
        self.grad_beta = np.zeros(features)
        self.running_mean = np.zeros(features)
        self.running_variance = np.ones(features)
        self.momentum = momentum
        self.epsilon = epsilon
        self._normalized = None
        self._inverse_std = None
        self._training = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.gamma.size:
            raise ValueError("x must have shape (nonempty batch, features)")
        if training:
            mean = x.mean(axis=0)
            variance = x.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_variance = (
                (1 - self.momentum) * self.running_variance + self.momentum * variance
            )
        else:
            mean = self.running_mean
            variance = self.running_variance
        self._inverse_std = 1 / np.sqrt(variance + self.epsilon)
        self._normalized = (x - mean) * self._inverse_std
        self._training = training
        return self.gamma * self._normalized + self.beta

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._normalized is None:
            raise RuntimeError("call forward before backward")
        grad_output = np.asarray(grad_output, dtype=float)
        if grad_output.shape != self._normalized.shape:
            raise ValueError("grad_output shape does not match forward output")
        self.grad_gamma = np.sum(grad_output * self._normalized, axis=0)
        self.grad_beta = np.sum(grad_output, axis=0)
        if not self._training:
            return grad_output * self.gamma * self._inverse_std

        count = grad_output.shape[0]
        scaled = grad_output * self.gamma
        return self._inverse_std / count * (
            count * scaled
            - np.sum(scaled, axis=0)
            - self._normalized * np.sum(scaled * self._normalized, axis=0)
        )

    def parameters_and_gradients(self):
        yield self.gamma, self.grad_gamma
        yield self.beta, self.grad_beta

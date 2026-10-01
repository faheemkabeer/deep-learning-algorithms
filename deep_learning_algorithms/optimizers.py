"""In-place stochastic gradient descent and Adam updates."""

from collections.abc import Iterable

import numpy as np


ParameterGradients = Iterable[tuple[np.ndarray, np.ndarray]]


class SGD:
    """Plain stochastic gradient descent, optionally with momentum."""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        if learning_rate <= 0 or not 0 <= momentum < 1:
            raise ValueError("learning_rate must be positive and momentum must be in [0, 1)")
        self.learning_rate = learning_rate
        self.momentum = momentum
        self._velocity = {}

    def step(self, parameters: ParameterGradients) -> None:
        for parameter, gradient in parameters:
            if parameter.shape != gradient.shape:
                raise ValueError("parameter and gradient shapes differ")
            key = id(parameter)
            velocity = self._velocity.setdefault(key, np.zeros_like(parameter))
            velocity *= self.momentum
            velocity += gradient
            parameter -= self.learning_rate * velocity


class Adam:
    """Adam with per-parameter first and second moments and bias correction."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        if learning_rate <= 0 or epsilon <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("invalid Adam hyperparameters")
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._state = {}

    def step(self, parameters: ParameterGradients) -> None:
        for parameter, gradient in parameters:
            if parameter.shape != gradient.shape:
                raise ValueError("parameter and gradient shapes differ")
            key = id(parameter)
            if key not in self._state:
                self._state[key] = [np.zeros_like(parameter), np.zeros_like(parameter), 0]
            first, second, count = self._state[key]
            count += 1
            first *= self.beta1
            first += (1 - self.beta1) * gradient
            second *= self.beta2
            second += (1 - self.beta2) * gradient**2
            corrected_first = first / (1 - self.beta1**count)
            corrected_second = second / (1 - self.beta2**count)
            parameter -= self.learning_rate * corrected_first / (np.sqrt(corrected_second) + self.epsilon)
            self._state[key][2] = count

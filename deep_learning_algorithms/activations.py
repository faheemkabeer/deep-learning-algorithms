"""Numerically stable activation functions."""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Apply rectified linear activation elementwise."""
    return np.maximum(np.asarray(x), 0)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU; choose zero at the non-differentiable origin."""
    return (np.asarray(x) > 0).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid without overflowing for large negative values."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1 / (1 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1 + exp_x)
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert logits to probabilities along one axis."""
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

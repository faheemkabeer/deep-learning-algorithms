"""Losses and gradients for batched training."""

import numpy as np

from .activations import softmax


def mean_squared_error(prediction: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Return mean squared error and its gradient with respect to prediction."""
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("prediction and target must have the same nonempty shape")
    difference = prediction - target
    return float(np.mean(difference**2)), 2 * difference / difference.size


def softmax_cross_entropy(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return average cross entropy and gradient for integer class labels.

    ``logits`` must have shape (batch, classes), ``labels`` shape (batch,).
    The loss uses log-sum-exp, so large logits do not overflow.
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have shape (nonempty batch, classes)")
    if labels.shape != (logits.shape[0],) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a batch of integer class indices")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("label index is out of range")

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_normalizer = np.log(np.sum(np.exp(shifted), axis=1))
    rows = np.arange(logits.shape[0])
    loss = np.mean(log_normalizer - shifted[rows, labels])

    gradient = softmax(logits, axis=1)
    gradient[rows, labels] -= 1
    gradient /= logits.shape[0]
    return float(loss), gradient

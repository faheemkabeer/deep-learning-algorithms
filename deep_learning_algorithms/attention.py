"""Scaled dot product attention, the core operation in Transformers."""

import numpy as np

from .activations import softmax


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return context vectors and attention weights.

    Arrays are (batch, tokens, features). ``mask`` broadcasts to
    (batch, query_tokens, key_tokens), where True means attendable.
    """
    query = np.asarray(query, dtype=float)
    key = np.asarray(key, dtype=float)
    value = np.asarray(value, dtype=float)
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key, and value must be three-dimensional")
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ValueError("batch dimensions must match")
    if query.shape[2] != key.shape[2] or key.shape[1] != value.shape[1]:
        raise ValueError("query/key features and key/value token counts must match")
    if query.shape[2] == 0 or key.shape[1] == 0:
        raise ValueError("feature and key-token dimensions must be nonempty")

    scores = query @ np.swapaxes(key, -1, -2) / np.sqrt(query.shape[-1])
    if mask is not None:
        try:
            allowed = np.broadcast_to(np.asarray(mask, dtype=bool), scores.shape)
        except ValueError as error:
            raise ValueError("mask is not broadcastable to attention scores") from error
        if not np.all(allowed.any(axis=-1)):
            raise ValueError("each query must have at least one unmasked key")
        scores = np.where(allowed, scores, -np.inf)

    weights = softmax(scores, axis=-1)
    return weights @ value, weights


def causal_mask(tokens: int) -> np.ndarray:
    """Make a lower-triangular mask so positions cannot see future tokens."""
    if tokens < 1:
        raise ValueError("tokens must be positive")
    return np.tril(np.ones((tokens, tokens), dtype=bool))


def scaled_dot_product_attention_backward(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    grad_context: np.ndarray,
    mask: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gradients for query, key, and value inputs.

    Uses the softmax Jacobian-vector product without constructing the full
    Jacobian. Masked key positions receive zero score gradient.
    """
    query = np.asarray(query, dtype=float)
    key = np.asarray(key, dtype=float)
    value = np.asarray(value, dtype=float)
    grad_context = np.asarray(grad_context, dtype=float)
    context, weights = scaled_dot_product_attention(query, key, value, mask)
    if grad_context.shape != context.shape:
        raise ValueError(f"grad_context must have shape {context.shape}")

    grad_weights = grad_context @ np.swapaxes(value, -1, -2)
    grad_value = np.swapaxes(weights, -1, -2) @ grad_context
    grad_scores = weights * (
        grad_weights - np.sum(grad_weights * weights, axis=-1, keepdims=True)
    )
    scale = np.sqrt(query.shape[-1])
    grad_query = grad_scores @ key / scale
    grad_key = np.swapaxes(grad_scores, -1, -2) @ query / scale
    return grad_query, grad_key, grad_value

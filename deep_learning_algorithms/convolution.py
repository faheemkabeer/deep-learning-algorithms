"""Reference implementation of 2D convolution for NCHW tensors."""

from __future__ import annotations

import numpy as np


def conv2d(
    x: np.ndarray,
    kernels: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Cross-correlate inputs with kernels, as deep learning libraries do.

    ``x``: (batch, in_channels, height, width).
    ``kernels``: (out_channels, in_channels, kernel_height, kernel_width).
    The reference loops emphasize indexing rather than performance.
    """
    x = np.asarray(x, dtype=float)
    kernels = np.asarray(kernels, dtype=float)
    if x.ndim != 4 or kernels.ndim != 4:
        raise ValueError("x and kernels must be four-dimensional")
    if x.shape[1] != kernels.shape[1] or not all(kernels.shape):
        raise ValueError("kernel channels or dimensions are invalid")
    if stride < 1 or padding < 0:
        raise ValueError("stride must be positive and padding nonnegative")
    if bias is None:
        bias = np.zeros(kernels.shape[0])
    bias = np.asarray(bias, dtype=float)
    if bias.shape != (kernels.shape[0],):
        raise ValueError("bias must have one value per output channel")

    kh, kw = kernels.shape[2:]
    padded_height = x.shape[2] + 2 * padding
    padded_width = x.shape[3] + 2 * padding
    if padded_height < kh or padded_width < kw:
        raise ValueError("kernel is larger than padded input")
    output_height = (padded_height - kh) // stride + 1
    output_width = (padded_width - kw) // stride + 1
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    output = np.empty((x.shape[0], kernels.shape[0], output_height, output_width))

    for row in range(output_height):
        for col in range(output_width):
            h, w = row * stride, col * stride
            patch = padded[:, :, h : h + kh, w : w + kw]
            output[:, :, row, col] = np.tensordot(
                patch, kernels, axes=([1, 2, 3], [1, 2, 3])
            ) + bias
    return output


def conv2d_backward(
    x: np.ndarray,
    kernels: np.ndarray,
    grad_output: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagate through ``conv2d``.

    Return gradients for input, kernels, and one bias per output channel.
    Gradients are summed over the batch; the upstream loss controls averaging.
    """
    x = np.asarray(x, dtype=float)
    kernels = np.asarray(kernels, dtype=float)
    grad_output = np.asarray(grad_output, dtype=float)
    expected = conv2d(x, kernels, stride=stride, padding=padding).shape
    if grad_output.shape != expected:
        raise ValueError(f"grad_output must have shape {expected}")

    kh, kw = kernels.shape[2:]
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    grad_padded = np.zeros_like(padded)
    grad_kernels = np.zeros_like(kernels)
    grad_bias = grad_output.sum(axis=(0, 2, 3))

    for row in range(grad_output.shape[2]):
        for col in range(grad_output.shape[3]):
            h, w = row * stride, col * stride
            patch = padded[:, :, h : h + kh, w : w + kw]
            upstream = grad_output[:, :, row, col]
            grad_kernels += np.tensordot(upstream, patch, axes=([0], [0]))
            grad_padded[:, :, h : h + kh, w : w + kw] += np.tensordot(
                upstream, kernels, axes=([1], [0])
            )

    if padding:
        grad_input = grad_padded[:, :, padding:-padding, padding:-padding]
    else:
        grad_input = grad_padded
    return grad_input, grad_kernels, grad_bias

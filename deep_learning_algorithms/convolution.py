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

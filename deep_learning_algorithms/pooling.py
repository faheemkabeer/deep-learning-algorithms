"""Max pooling and its backward pass for NCHW image tensors."""

import numpy as np


def _pool_shape(x: np.ndarray, kernel_size: int, stride: int) -> tuple[int, int]:
    if x.ndim != 4:
        raise ValueError("x must have shape (batch, channels, height, width)")
    if kernel_size < 1 or stride < 1:
        raise ValueError("kernel_size and stride must be positive")
    if kernel_size > x.shape[2] or kernel_size > x.shape[3]:
        raise ValueError("pooling window is larger than the input")
    return (
        (x.shape[2] - kernel_size) // stride + 1,
        (x.shape[3] - kernel_size) // stride + 1,
    )


def max_pool2d(x: np.ndarray, kernel_size: int = 2, stride: int = None) -> np.ndarray:
    """Take the maximum in each spatial window; ties use the first element."""
    x = np.asarray(x, dtype=float)
    if stride is None:
        stride = kernel_size
    output_height, output_width = _pool_shape(x, kernel_size, stride)
    output = np.empty((x.shape[0], x.shape[1], output_height, output_width))
    for row in range(output_height):
        for col in range(output_width):
            h, w = row * stride, col * stride
            patch = x[:, :, h : h + kernel_size, w : w + kernel_size]
            output[:, :, row, col] = patch.max(axis=(2, 3))
    return output


def max_pool2d_backward(
    x: np.ndarray,
    grad_output: np.ndarray,
    kernel_size: int = 2,
    stride: int = None,
) -> np.ndarray:
    """Route each upstream gradient to the winning input cell.

    Gradients from overlapping windows accumulate. When values tie, the first
    cell in row-major order receives the gradient.
    """
    x = np.asarray(x, dtype=float)
    grad_output = np.asarray(grad_output, dtype=float)
    if stride is None:
        stride = kernel_size
    output_height, output_width = _pool_shape(x, kernel_size, stride)
    expected = (x.shape[0], x.shape[1], output_height, output_width)
    if grad_output.shape != expected:
        raise ValueError(f"grad_output must have shape {expected}")

    grad_input = np.zeros_like(x)
    batch_index, channel_index = np.indices(x.shape[:2])
    for row in range(output_height):
        for col in range(output_width):
            h, w = row * stride, col * stride
            patch = x[:, :, h : h + kernel_size, w : w + kernel_size]
            winner = patch.reshape(*x.shape[:2], -1).argmax(axis=-1)
            grad_input[
                batch_index,
                channel_index,
                h + winner // kernel_size,
                w + winner % kernel_size,
            ] += grad_output[:, :, row, col]
    return grad_input

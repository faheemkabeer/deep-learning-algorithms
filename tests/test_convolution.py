import unittest

import numpy as np

from deep_learning_algorithms.convolution import conv2d, conv2d_backward


class ConvolutionTests(unittest.TestCase):
    def test_valid_convolution(self):
        x = np.arange(1, 10, dtype=float).reshape(1, 1, 3, 3)
        kernel = np.array([[[[1, 0], [0, -1]]]], dtype=float)
        expected = np.full((1, 1, 2, 2), -4.0)
        np.testing.assert_allclose(conv2d(x, kernel), expected)

    def test_padding_stride_and_channels(self):
        x = np.ones((2, 2, 3, 3))
        kernels = np.ones((3, 2, 2, 2))
        output = conv2d(x, kernels, bias=np.array([0, 1, 2]), stride=2, padding=1)
        self.assertEqual(output.shape, (2, 3, 2, 2))
        np.testing.assert_allclose(output[:, :, 0, 0], [[2, 3, 4], [2, 3, 4]])

    def test_mismatched_channels(self):
        with self.assertRaises(ValueError):
            conv2d(np.ones((1, 2, 3, 3)), np.ones((1, 1, 2, 2)))

    def test_backward_matches_finite_difference(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(1, 2, 3, 3))
        kernels = rng.normal(size=(2, 2, 2, 2))
        bias = np.array([0.2, -0.3])
        upstream = rng.normal(size=(1, 2, 2, 2))
        grad_input, grad_kernels, grad_bias = conv2d_backward(x, kernels, upstream)

        def objective():
            return float(np.sum(conv2d(x, kernels, bias) * upstream))

        epsilon = 1e-6
        for parameter, gradient in ((x, grad_input), (kernels, grad_kernels), (bias, grad_bias)):
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus = objective()
                parameter[index] = original - epsilon
                minus = objective()
                parameter[index] = original
                self.assertAlmostEqual(gradient[index], (plus - minus) / (2 * epsilon), places=6)

    def test_backward_with_stride_and_padding(self):
        x = np.ones((1, 1, 3, 3))
        kernels = np.ones((1, 1, 2, 2))
        upstream = np.ones((1, 1, 2, 2))
        grad_input, grad_kernels, grad_bias = conv2d_backward(
            x, kernels, upstream, stride=2, padding=1
        )
        self.assertEqual(grad_input.shape, x.shape)
        self.assertEqual(grad_kernels.shape, kernels.shape)
        np.testing.assert_allclose(grad_bias, [4])
        with self.assertRaises(ValueError):
            conv2d_backward(x, kernels, np.ones((1, 1, 1, 1)), stride=2, padding=1)


if __name__ == "__main__":
    unittest.main()

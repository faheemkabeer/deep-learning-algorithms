import unittest

import numpy as np

from deep_learning_algorithms.convolution import conv2d


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


if __name__ == "__main__":
    unittest.main()

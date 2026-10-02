import unittest

import numpy as np

from deep_learning_algorithms.pooling import max_pool2d, max_pool2d_backward


class PoolingTests(unittest.TestCase):
    def test_forward_and_backward(self):
        x = np.array([[[[1, 4, 2, 3], [3, 2, 8, 5], [0, 7, 1, 2], [6, 1, 4, 9]]]], dtype=float)
        output = max_pool2d(x)
        np.testing.assert_array_equal(output, [[[[4, 8], [7, 9]]]])
        gradient = max_pool2d_backward(x, np.ones_like(output))
        expected = np.zeros_like(x)
        expected[0, 0, 0, 1] = 1
        expected[0, 0, 1, 2] = 1
        expected[0, 0, 2, 1] = 1
        expected[0, 0, 3, 3] = 1
        np.testing.assert_array_equal(gradient, expected)

    def test_overlapping_windows_accumulate(self):
        x = np.zeros((1, 1, 3, 3))
        x[0, 0, 1, 1] = 10
        output = max_pool2d(x, kernel_size=2, stride=1)
        np.testing.assert_array_equal(output, np.full((1, 1, 2, 2), 10))
        gradient = max_pool2d_backward(x, np.ones_like(output), kernel_size=2, stride=1)
        self.assertEqual(gradient[0, 0, 1, 1], 4)
        self.assertEqual(gradient.sum(), 4)

    def test_invalid_upstream_shape(self):
        with self.assertRaises(ValueError):
            max_pool2d_backward(np.ones((1, 1, 4, 4)), np.ones((1, 1, 1, 1)))


if __name__ == "__main__":
    unittest.main()

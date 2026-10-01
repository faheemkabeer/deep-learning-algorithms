import unittest

import numpy as np

from deep_learning_algorithms.activations import relu, relu_derivative, sigmoid, softmax
from deep_learning_algorithms.losses import mean_squared_error, softmax_cross_entropy


class ActivationLossTests(unittest.TestCase):
    def test_stable_sigmoid_and_softmax(self):
        np.testing.assert_allclose(sigmoid(np.array([-1000.0, 0.0, 1000.0])), [0, 0.5, 1])
        probabilities = softmax(np.array([[1000.0, 1001.0], [-1000.0, -999.0]]))
        np.testing.assert_allclose(probabilities.sum(axis=1), [1, 1])
        self.assertTrue(np.isfinite(probabilities).all())

    def test_relu_and_mse(self):
        np.testing.assert_array_equal(relu(np.array([-2, 0, 3])), [0, 0, 3])
        np.testing.assert_array_equal(relu_derivative(np.array([-2, 0, 3])), [0, 0, 1])
        loss, gradient = mean_squared_error(np.array([1.0, 3.0]), np.array([0.0, 1.0]))
        self.assertAlmostEqual(loss, 2.5)
        np.testing.assert_allclose(gradient, [1, 2])

    def test_cross_entropy_gradient_against_finite_difference(self):
        logits = np.array([[0.2, -0.4, 1.1], [1.0, 0.8, -0.3]])
        labels = np.array([2, 1])
        loss, gradient = softmax_cross_entropy(logits, labels)
        self.assertGreater(loss, 0)
        numerical = np.empty_like(logits)
        epsilon = 1e-6
        for index in np.ndindex(logits.shape):
            plus, minus = logits.copy(), logits.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numerical[index] = (
                softmax_cross_entropy(plus, labels)[0]
                - softmax_cross_entropy(minus, labels)[0]
            ) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical, atol=1e-8)

    def test_invalid_labels(self):
        with self.assertRaises(ValueError):
            softmax_cross_entropy(np.zeros((2, 2)), np.array([0, 2]))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from deep_learning_algorithms.layers import Dense, MLP
from deep_learning_algorithms.losses import softmax_cross_entropy


class LayerTests(unittest.TestCase):
    def test_dense_gradient_against_finite_difference(self):
        layer = Dense(2, 3, np.random.default_rng(3))
        x = np.array([[0.2, 0.7], [-0.4, 1.1]])
        upstream = np.array([[1.0, -0.5, 0.1], [0.3, 0.2, -0.6]])
        layer.forward(x)
        grad_input = layer.backward(upstream)

        def objective():
            return float(np.sum(layer.forward(x) * upstream))

        epsilon = 1e-6
        for parameter, gradient in layer.parameters_and_gradients():
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus = objective()
                parameter[index] = original - epsilon
                minus = objective()
                parameter[index] = original
                self.assertAlmostEqual(gradient[index], (plus - minus) / (2 * epsilon), places=7)
        self.assertEqual(grad_input.shape, x.shape)

    def test_mlp_backpropagation(self):
        network = MLP([2, 5, 2], seed=8)
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        logits = network.forward(x)
        loss, grad = softmax_cross_entropy(logits, np.array([0, 1, 1, 0]))
        grad_input = network.backward(grad)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(logits.shape, (4, 2))
        self.assertEqual(grad_input.shape, x.shape)
        self.assertEqual(len(list(network.parameters_and_gradients())), 4)

    def test_invalid_shapes(self):
        layer = Dense(2, 3, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            layer.forward(np.zeros((4, 4)))
        with self.assertRaises(RuntimeError):
            layer.backward(np.zeros((4, 3)))


if __name__ == "__main__":
    unittest.main()

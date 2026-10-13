import unittest

import numpy as np

from deep_learning_algorithms.recurrent import GRU, LSTM


class LSTMTests(unittest.TestCase):
    def test_shapes_and_parameter_interface(self):
        layer = LSTM(3, 4, seed=7)
        output = layer.forward(np.ones((2, 5, 3)))
        self.assertEqual(output.shape, (2, 5, 4))
        self.assertEqual([p.shape for p, _ in layer.parameters_and_gradients()], [(7, 16), (16,)])
        self.assertEqual(layer.backward(np.ones_like(output)).shape, (2, 5, 3))

    def test_backward_matches_finite_difference(self):
        rng = np.random.default_rng(4)
        layer = LSTM(2, 2, seed=8)
        x = rng.normal(size=(1, 3, 2))
        upstream = rng.normal(size=(1, 3, 2))
        layer.forward(x)
        analytic_x = layer.backward(upstream)
        epsilon = 1e-5
        for index in np.ndindex(x.shape):
            original = x[index]
            x[index] = original + epsilon
            plus = np.sum(layer.forward(x) * upstream)
            x[index] = original - epsilon
            minus = np.sum(layer.forward(x) * upstream)
            x[index] = original
            self.assertAlmostEqual(analytic_x[index], (plus - minus) / (2 * epsilon), places=5)

    def test_rejects_wrong_shapes(self):
        layer = LSTM(2, 3)
        with self.assertRaises(ValueError):
            layer.forward(np.zeros((2, 2)))
        with self.assertRaises(RuntimeError):
            layer.backward(np.zeros((1, 1, 3)))


class GRUTests(unittest.TestCase):
    def test_shapes_and_parameter_interface(self):
        layer = GRU(3, 4, seed=7)
        output = layer.forward(np.ones((2, 5, 3)))
        self.assertEqual(output.shape, (2, 5, 4))
        self.assertEqual([p.shape for p, _ in layer.parameters_and_gradients()], [(7, 12), (12,)])
        self.assertEqual(layer.backward(np.ones_like(output)).shape, (2, 5, 3))

    def test_backward_matches_finite_difference(self):
        rng = np.random.default_rng(9)
        layer = GRU(2, 2, seed=11)
        x = rng.normal(size=(1, 3, 2))
        upstream = rng.normal(size=(1, 3, 2))
        layer.forward(x)
        analytic_x = layer.backward(upstream)
        epsilon = 1e-5
        for index in np.ndindex(x.shape):
            original = x[index]
            x[index] = original + epsilon
            plus = np.sum(layer.forward(x) * upstream)
            x[index] = original - epsilon
            minus = np.sum(layer.forward(x) * upstream)
            x[index] = original
            self.assertAlmostEqual(analytic_x[index], (plus - minus) / (2 * epsilon), places=5)


if __name__ == "__main__":
    unittest.main()

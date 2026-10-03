import unittest

import numpy as np

from deep_learning_algorithms.normalization import BatchNorm1d


class BatchNormTests(unittest.TestCase):
    def test_training_normalizes_and_eval_uses_running_statistics(self):
        layer = BatchNorm1d(2, momentum=1.0)
        batch = np.array([[1.0, 10.0], [2.0, 12.0], [3.0, 14.0]])
        normalized = layer.forward(batch)
        np.testing.assert_allclose(normalized.mean(axis=0), [0, 0], atol=1e-12)
        np.testing.assert_allclose(layer.running_mean, [2, 12])
        before = layer.running_mean.copy()
        evaluated = layer.forward(batch + 1, training=False)
        np.testing.assert_allclose(layer.running_mean, before)
        self.assertEqual(evaluated.shape, batch.shape)

    def test_gradients_match_finite_difference(self):
        rng = np.random.default_rng(15)
        x = rng.normal(size=(4, 3))
        upstream = rng.normal(size=(4, 3))
        layer = BatchNorm1d(3)
        layer.forward(x)
        grad_input = layer.backward(upstream)

        def objective():
            return float(np.sum(layer.forward(x) * upstream))

        epsilon = 1e-6
        for parameter, gradient in (
            (x, grad_input),
            (layer.gamma, layer.grad_gamma),
            (layer.beta, layer.grad_beta),
        ):
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus = objective()
                parameter[index] = original - epsilon
                minus = objective()
                parameter[index] = original
                self.assertAlmostEqual(gradient[index], (plus - minus) / (2 * epsilon), places=6)

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            BatchNorm1d(2).forward(np.ones((3, 4)))


if __name__ == "__main__":
    unittest.main()

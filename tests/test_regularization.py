import unittest

import numpy as np

from deep_learning_algorithms.layers import MLP
from deep_learning_algorithms.regularization import Dropout


class DropoutTests(unittest.TestCase):
    def test_training_preserves_scale_in_expectation(self):
        layer = Dropout(probability=0.25, seed=5)
        values = np.ones(20000)
        output = layer.forward(values, training=True)
        self.assertAlmostEqual(output.mean(), 1.0, delta=0.02)
        np.testing.assert_array_equal(layer.backward(np.ones_like(values)), output)

    def test_evaluation_is_identity(self):
        layer = Dropout(probability=0.5, seed=1)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(layer.forward(x, training=False), x)
        np.testing.assert_array_equal(layer.backward(np.ones_like(x)), np.ones_like(x))

    def test_mlp_training_and_evaluation(self):
        network = MLP([2, 8, 2], seed=4, dropout=0.5)
        x = np.ones((8, 2))
        training_output = network.forward(x, training=True)
        self.assertEqual(training_output.shape, (8, 2))
        self.assertEqual(network.backward(np.ones_like(training_output)).shape, x.shape)
        evaluation_output = network.forward(x, training=False)
        np.testing.assert_allclose(network.forward(x, training=False), evaluation_output)

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            Dropout(1.0)


if __name__ == "__main__":
    unittest.main()

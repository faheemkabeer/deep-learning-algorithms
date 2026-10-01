import unittest

import numpy as np

from deep_learning_algorithms.optimizers import Adam, SGD


class OptimizerTests(unittest.TestCase):
    def test_sgd_single_step(self):
        parameter = np.array([1.0, -1.0])
        SGD(learning_rate=0.1).step([(parameter, np.array([2.0, -4.0]))])
        np.testing.assert_allclose(parameter, [0.8, -0.6])

    def test_adam_converges_on_quadratic(self):
        parameter = np.array([5.0])
        optimizer = Adam(learning_rate=0.1)
        for _ in range(200):
            optimizer.step([(parameter, 2 * (parameter - 2))])
        self.assertAlmostEqual(parameter[0], 2, places=3)

    def test_separate_parameter_state(self):
        a = np.array([2.0])
        b = np.array([-2.0])
        optimizer = Adam(learning_rate=0.1)
        optimizer.step([(a, np.array([1.0])), (b, np.array([-1.0]))])
        np.testing.assert_allclose(a, [1.9], atol=1e-6)
        np.testing.assert_allclose(b, [-1.9], atol=1e-6)


if __name__ == "__main__":
    unittest.main()

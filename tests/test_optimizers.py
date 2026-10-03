import unittest

import numpy as np

from deep_learning_algorithms.optimizers import Adam, SGD, clip_grad_norm


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

    def test_global_gradient_clipping(self):
        first = np.zeros(1)
        second = np.zeros(1)
        first_gradient = np.array([3.0])
        second_gradient = np.array([4.0])
        original_norm = clip_grad_norm(
            [(first, first_gradient), (second, second_gradient)], max_norm=2.5
        )
        self.assertEqual(original_norm, 5.0)
        np.testing.assert_allclose(first_gradient, [1.5])
        np.testing.assert_allclose(second_gradient, [2.0])

    def test_clipping_noop_and_invalid_gradient(self):
        parameter = np.zeros(2)
        gradient = np.array([0.1, 0.2])
        before = gradient.copy()
        self.assertLess(clip_grad_norm([(parameter, gradient)], max_norm=1), 1)
        np.testing.assert_array_equal(gradient, before)
        with self.assertRaises(ValueError):
            clip_grad_norm([(parameter, np.array([np.inf, 0.0]))], max_norm=1)


if __name__ == "__main__":
    unittest.main()

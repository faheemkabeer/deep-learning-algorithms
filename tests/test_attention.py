import unittest

import numpy as np

from deep_learning_algorithms.attention import (
    causal_mask,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
)


class AttentionTests(unittest.TestCase):
    def test_uniform_attention(self):
        query = np.zeros((1, 1, 2))
        key = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        value = np.array([[[2.0], [4.0]]])
        context, weights = scaled_dot_product_attention(query, key, value)
        np.testing.assert_allclose(weights, [[[0.5, 0.5]]])
        np.testing.assert_allclose(context, [[[3.0]]])

    def test_causal_mask_excludes_future(self):
        vectors = np.eye(3)[None, :, :]
        context, weights = scaled_dot_product_attention(
            vectors, vectors, vectors, mask=causal_mask(3)
        )
        self.assertEqual(context.shape, (1, 3, 3))
        np.testing.assert_allclose(weights[0, 0], [1, 0, 0])
        self.assertEqual(weights[0, 1, 2], 0)
        np.testing.assert_allclose(weights.sum(axis=-1), 1)

    def test_reject_all_masked_row(self):
        vectors = np.ones((1, 2, 2))
        with self.assertRaises(ValueError):
            scaled_dot_product_attention(vectors, vectors, vectors, mask=np.zeros((2, 2)))

    def test_backward_matches_finite_difference(self):
        rng = np.random.default_rng(12)
        query = rng.normal(size=(2, 2, 3))
        key = rng.normal(size=(2, 3, 3))
        value = rng.normal(size=(2, 3, 2))
        upstream = rng.normal(size=(2, 2, 2))
        gradients = scaled_dot_product_attention_backward(query, key, value, upstream)

        def objective():
            context, _ = scaled_dot_product_attention(query, key, value)
            return float(np.sum(context * upstream))

        epsilon = 1e-6
        for parameter, gradient in zip((query, key, value), gradients):
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus = objective()
                parameter[index] = original - epsilon
                minus = objective()
                parameter[index] = original
                self.assertAlmostEqual(gradient[index], (plus - minus) / (2 * epsilon), places=6)

    def test_masked_gradient(self):
        vectors = np.eye(3)[None, :, :]
        upstream = np.ones_like(vectors)
        grad_query, grad_key, grad_value = scaled_dot_product_attention_backward(
            vectors, vectors, vectors, upstream, mask=causal_mask(3)
        )
        self.assertEqual(grad_query.shape, vectors.shape)
        self.assertEqual(grad_key.shape, vectors.shape)
        self.assertEqual(grad_value.shape, vectors.shape)
        np.testing.assert_allclose(grad_query[0, 0], 0)
        self.assertTrue(np.isfinite(grad_value).all())


if __name__ == "__main__":
    unittest.main()

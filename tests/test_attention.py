import unittest

import numpy as np

from deep_learning_algorithms.attention import causal_mask, scaled_dot_product_attention


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


if __name__ == "__main__":
    unittest.main()

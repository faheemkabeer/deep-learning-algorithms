import unittest

import numpy as np

from examples.train_xor import train_xor


class TrainingExampleTests(unittest.TestCase):
    def test_xor_training(self):
        model, loss = train_xor()
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        predictions = model.forward(inputs).argmax(axis=1)
        np.testing.assert_array_equal(predictions, [0, 1, 1, 0])
        self.assertLess(loss, 0.05)


if __name__ == "__main__":
    unittest.main()

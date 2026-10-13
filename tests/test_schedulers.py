import unittest

from deep_learning_algorithms.schedulers import CosineDecay, ExponentialDecay, LinearWarmup, StepDecay


class SchedulerTests(unittest.TestCase):
    def test_step_decay(self):
        schedule = StepDecay(0.1, step_size=3, decay=0.5)
        self.assertAlmostEqual(schedule(0), 0.1)
        self.assertAlmostEqual(schedule(2), 0.1)
        self.assertAlmostEqual(schedule(3), 0.05)
        self.assertAlmostEqual(schedule(6), 0.025)

    def test_exponential_decay(self):
        schedule = ExponentialDecay(0.1, decay_rate=0.9)
        self.assertAlmostEqual(schedule(0), 0.1)
        self.assertAlmostEqual(schedule(2), 0.081)

    def test_cosine_decay_clamps_at_minimum(self):
        schedule = CosineDecay(0.1, total_steps=4, minimum_rate=0.01)
        self.assertAlmostEqual(schedule(0), 0.1)
        self.assertAlmostEqual(schedule(2), 0.055)
        self.assertAlmostEqual(schedule(4), 0.01)
        self.assertAlmostEqual(schedule(99), 0.01)

    def test_linear_warmup_clamps_at_target(self):
        schedule = LinearWarmup(0.1, warmup_steps=4, start_rate=0.02)
        self.assertAlmostEqual(schedule(0), 0.02)
        self.assertAlmostEqual(schedule(2), 0.06)
        self.assertAlmostEqual(schedule(4), 0.1)
        self.assertAlmostEqual(schedule(10), 0.1)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            StepDecay(0.1, 0)
        with self.assertRaises(ValueError):
            ExponentialDecay(0.1, 1.1)
        with self.assertRaises(ValueError):
            CosineDecay(0.1, 1, minimum_rate=0.2)
        with self.assertRaises(ValueError):
            LinearWarmup(0.1, 1, start_rate=-1)
        with self.assertRaises(ValueError):
            ExponentialDecay(0.1)(-1)


if __name__ == "__main__":
    unittest.main()

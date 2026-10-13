"""Learning-rate schedules for reproducible optimizer training loops."""

import math


def _validate_rate(learning_rate: float) -> float:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    return float(learning_rate)


def _validate_step(step: int) -> int:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    return step


class StepDecay:
    """Multiply the learning rate by ``decay`` every ``step_size`` steps."""

    def __init__(self, learning_rate: float, step_size: int, decay: float = 0.1):
        self.learning_rate = _validate_rate(learning_rate)
        if step_size < 1 or not 0 < decay <= 1:
            raise ValueError("step_size must be positive and decay must be in (0, 1]")
        self.step_size = step_size
        self.decay = float(decay)

    def __call__(self, step: int) -> float:
        return self.learning_rate * self.decay ** (_validate_step(step) // self.step_size)


class ExponentialDecay:
    """Exponentially decay the learning rate once per training step."""

    def __init__(self, learning_rate: float, decay_rate: float = 0.96):
        self.learning_rate = _validate_rate(learning_rate)
        if not 0 < decay_rate <= 1:
            raise ValueError("decay_rate must be in (0, 1]")
        self.decay_rate = float(decay_rate)

    def __call__(self, step: int) -> float:
        return self.learning_rate * self.decay_rate ** _validate_step(step)


class CosineDecay:
    """Cosine annealing from ``learning_rate`` to ``minimum_rate``."""

    def __init__(self, learning_rate: float, total_steps: int, minimum_rate: float = 0.0):
        self.learning_rate = _validate_rate(learning_rate)
        if total_steps < 1 or minimum_rate < 0 or minimum_rate > learning_rate:
            raise ValueError("total_steps must be positive and minimum_rate must be in [0, learning_rate]")
        self.total_steps = total_steps
        self.minimum_rate = float(minimum_rate)

    def __call__(self, step: int) -> float:
        progress = min(_validate_step(step), self.total_steps) / self.total_steps
        weight = 0.5 * (1 + math.cos(math.pi * progress))
        return self.minimum_rate + (self.learning_rate - self.minimum_rate) * weight


class LinearWarmup:
    """Linearly increase from ``start_rate`` to ``target_rate``."""

    def __init__(self, target_rate: float, warmup_steps: int, start_rate: float = 0.0):
        self.target_rate = _validate_rate(target_rate)
        if warmup_steps < 1 or start_rate < 0 or start_rate > target_rate:
            raise ValueError("warmup_steps must be positive and start_rate must be in [0, target_rate]")
        self.warmup_steps = warmup_steps
        self.start_rate = float(start_rate)

    def __call__(self, step: int) -> float:
        progress = min(_validate_step(step), self.warmup_steps) / self.warmup_steps
        return self.start_rate + (self.target_rate - self.start_rate) * progress

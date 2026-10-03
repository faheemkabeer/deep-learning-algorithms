"""Train a tiny MLP on XOR using the algorithms in this repository."""

from pathlib import Path
import sys

import numpy as np

# Allow ``python examples/train_xor.py`` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_learning_algorithms.activations import softmax
from deep_learning_algorithms.layers import MLP
from deep_learning_algorithms.losses import softmax_cross_entropy
from deep_learning_algorithms.optimizers import Adam, clip_grad_norm


def train_xor(steps: int = 1000) -> tuple[MLP, float]:
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    labels = np.array([0, 1, 1, 0])
    network = MLP([2, 8, 2], seed=7)
    optimizer = Adam(learning_rate=0.03)

    for _ in range(steps):
        logits = network.forward(inputs)
        _, grad_logits = softmax_cross_entropy(logits, labels)
        network.backward(grad_logits)
        clip_grad_norm(network.parameters_and_gradients(), max_norm=1.0)
        optimizer.step(network.parameters_and_gradients())

    logits = network.forward(inputs)
    loss, _ = softmax_cross_entropy(logits, labels)
    return network, loss


if __name__ == "__main__":
    model, final_loss = train_xor()
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    probabilities = softmax(model.forward(inputs))
    for sample, scores in zip(inputs.astype(int), probabilities):
        print(f"{sample.tolist()} -> class {scores.argmax()} (probability {scores.max():.3f})")
    print(f"Final cross-entropy: {final_loss:.4f}")

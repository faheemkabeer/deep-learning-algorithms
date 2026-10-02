# Deep Learning Algorithms from Scratch

Small, readable NumPy implementations of core deep learning building blocks. The repository is for learning and experimentation; it does not depend on a training framework.

## Contents

- Activation functions and losses
- Dense layers and backpropagation
- SGD and Adam optimizers
- 2D convolution with input, filter, and bias gradients
- Max pooling with backward gradient routing
- Scaled dot product attention with query, key, and value gradients
- XOR training example and unit tests

## Setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/train_xor.py
```

Inputs use NumPy arrays. Dense layers accept batches shaped `(batch, features)`. Convolution accepts NCHW arrays `(batch, channels, height, width)`. Attention accepts `(batch, tokens, features)`.

The algorithms prioritize clarity over speed. See each module's docstrings for shape and numerical details.

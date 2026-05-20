# funygrad

A tinygrad-inspired deep learning framework built from scratch in Python (with C/GPU backends planned).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

```python
from funygrad import Tensor, nn, optim

# Build a 2-layer MLP
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# Train with autograd
x = Tensor.randn(32, 784)
out = model(x)
loss = out.sum()
loss.backward()
```

## Features (Phase 1)

| Component | Status |
|-----------|--------|
| Lazy IR (LazyBuffer + Ops) | Done |
| Autograd engine | Done |
| NumPy backend | Done |
| nn.Module, Linear, ReLU, Sigmoid, Tanh | Done |
| CrossEntropyLoss, MSELoss | Done |
| SGD optimizer (momentum, weight decay) | Done |
| DataLoader, Dataset | Done |
| MNIST: 97.97% accuracy | Done |

## Examples

```bash
python3 examples/mnist.py       # MNIST training (downloads data automatically)
python3 examples/use_cases.py   # Tensor ops, autograd, regression, XOR
```

## Run Tests

```bash
python3 tests/test_ops.py
python3 tests/test_tensor.py
python3 tests/test_autograd.py
python3 tests/test_nn.py
python3 tests/test_optim.py
```

## Planned (Phase 2+)

- C/CPU codegen backend
- CUDA backend
- OpenCL backend
- Kernel fusion scheduler
- Conv2d, BatchNorm, Dropout
- Adam optimizer

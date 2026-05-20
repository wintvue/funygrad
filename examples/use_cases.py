"""
funygrad examples: basic operations, autograd, regression, XOR classification.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from funygrad import Tensor, nn, optim
from funygrad.data import TensorDataset, DataLoader

# ──────────────────────────────────────────────────
# 1. BASIC TENSOR OPERATIONS
# ──────────────────────────────────────────────────
print("=" * 50)
print("1. Basic Tensor Operations")
print("=" * 50)

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  a + b = {a + b}")
print(f"  a * b = {a * b}")
print(f"  dot(a,b) = {(a * b).sum()}")  # dot product
print(f"  a.relu() = {a.relu()}")
print(f"  a.sum() = {a.sum()}")
print(f"  a.mean() = {a.mean()}")

x = Tensor.randn(2, 3)
w = Tensor.randn(4, 3)
print(f"  randn(2,3) @ randn(4,3).T shape: {(x @ w.T).shape}")

# ──────────────────────────────────────────────────
# 2. AUTOGRAD
# ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("2. Autograd")
print("=" * 50)

# f(x) = x^2 + 3x, df/dx = 2x + 3
x = Tensor([2.0], requires_grad=True)
y = x * x + 3 * x
y.sum().backward()
print(f"  f(x) = x^2 + 3x,  x=2  =>  f(2)=10")
print(f"  df/dx = 2x + 3 = 2*2+3 = 7")
print(f"  computed grad: {x.grad.item():.1f}")

# Gradient of matmul: y = x @ W, dy/dx = W^T
x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
W = Tensor(np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32))
y = (x @ W).sum()
y.backward()
print(f"  x @ [[2,0],[0,2]], sum")
print(f"  dy/dx (expected = W^T = same):")
print(f"  {x.grad.numpy()}")

# ──────────────────────────────────────────────────
# 3. LINEAR REGRESSION
# ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("3. Linear Regression (y = 3*x1 + 2*x2 + 1)")
print("=" * 50)

np.random.seed(0)
n_samples = 200
X_data = np.random.randn(n_samples, 2).astype(np.float32)
y_data = (
    3 * X_data[:, 0] + 2 * X_data[:, 1] + 1 + 0.1 * np.random.randn(n_samples)
).astype(np.float32)
y_data = y_data.reshape(-1, 1)

X = Tensor(X_data)
y = Tensor(y_data)

model = nn.Linear(2, 1)
opt = optim.SGD(list(model.parameters()), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(200):
    opt.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    opt.step()

w_vals = model.weight.numpy().flatten()
b_val = model.bias.numpy().item()
print(f"  True:    y = 3.0*x1 + 2.0*x2 + 1.0")
print(f"  Learned: y = {w_vals[0]:.3f}*x1 + {w_vals[1]:.3f}*x2 + {b_val:.3f}")
print(f"  Final loss: {loss.item():.6f}")

# ──────────────────────────────────────────────────
# 4. XOR CLASSIFICATION
# ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("4. XOR Classification (2-hidden-layer MLP)")
print("=" * 50)

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_xor = np.array([0, 1, 1, 0], dtype=np.int64)


class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.fc1(x).tanh()
        x = self.fc2(x)
        return x


model_xor = XORNet()
opt_xor = optim.SGD(list(model_xor.parameters()), lr=0.5, momentum=0.9)
criterion = nn.CrossEntropyLoss()

X_t = Tensor(X_xor)
y_t = Tensor(y_xor)

for epoch in range(200):
    opt_xor.zero_grad()
    out = model_xor(X_t)
    loss = criterion(out, y_t)
    loss.backward()
    opt_xor.step()

preds = model_xor(X_t).numpy().argmax(axis=-1)
print(f"  Predictions: {preds.tolist()}")
print(f"  Ground truth: {y_xor.tolist()}")
print(f"  Accuracy: {(preds == y_xor).mean():.0%}")
print(f"  Final loss: {loss.item():.6f}")

# ──────────────────────────────────────────────────
# 5. SAVE / LOAD MODEL
# ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("5. Save / Load State Dict")
print("=" * 50)

state = model_xor.state_dict()
print(f"  State dict keys: {list(state.keys())}")
print(f"  fc1.weight shape: {state['fc1.weight'].shape}")

# Create new model and load
model2 = XORNet()
model2.load_state_dict(state)
out2 = model2(X_t)
print(f"  Loaded model predictions: {out2.numpy().argmax(axis=-1).tolist()}")
print(
    f"  Weights match: {np.allclose(model_xor.fc1.weight.numpy(), model2.fc1.weight.numpy())}"
)

print("\n" + "=" * 50)
print("All examples complete!")
print("=" * 50)

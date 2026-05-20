"""
MNIST training example using funygrad.

Trains a 2-layer MLP (784 -> 128 -> 10) on MNIST.
Downloads MNIST data automatically if not present.
Target: >90% accuracy within 10 epochs.
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gzip
import urllib.request
import numpy as np

from funygrad import Tensor, nn, optim
from funygrad.data import TensorDataset, DataLoader

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, path: str):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  Downloading {os.path.basename(path)}...")
    urllib.request.urlretrieve(url, path)


def _load_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        assert magic == 2051, f"Bad magic: {magic}"
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    return data.astype(np.float32) / 255.0


def _load_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        assert magic == 2049, f"Bad magic: {magic}"
        n = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def load_mnist(data_dir: str = "/tmp/funygrad_mnist"):
    """Download and load MNIST. Returns (X_train, y_train, X_test, y_test)."""
    paths = {}
    for name, url in MNIST_URLS.items():
        filename = url.split("/")[-1]
        path = os.path.join(data_dir, filename)
        paths[name] = path
        _download(url, path)

    X_train = _load_images(paths["train_images"])
    y_train = _load_labels(paths["train_labels"])
    X_test = _load_images(paths["test_images"])
    y_test = _load_labels(paths["test_labels"])
    return X_train, y_train, X_test, y_test


class MLP(nn.Module):
    def __init__(
        self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def accuracy(logits: Tensor, targets: np.ndarray) -> float:
    preds = logits.numpy().argmax(axis=-1)
    return (preds == targets).mean()


def main():
    epochs = 10
    batch_size = 64
    lr = 0.1

    print("Loading MNIST data...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.parameters()), lr=lr, momentum=0.9)
    print(f"Model: {model}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            x = Tensor(X_batch)
            target = Tensor(y_batch)

            out = model(x)
            loss = criterion(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        model.eval()
        n_correct = 0
        n_total = 0
        for X_batch, y_batch in test_loader:
            x = Tensor(X_batch)
            out = model(x)
            preds = out.numpy().argmax(axis=-1)
            n_correct += (preds == y_batch).sum()
            n_total += len(y_batch)
        test_acc = n_correct / n_total

        print(
            f"Epoch {epoch+1:2d}/{epochs}  loss: {avg_loss:.4f}  test_acc: {test_acc:.4f}"
        )

    print("\nTraining complete!")
    final_acc = test_acc
    assert final_acc >= 0.90, f"Expected >= 90% accuracy, got {final_acc:.2%}"
    print(f"Final test accuracy: {final_acc:.2%} -- PASSED (>= 90%)")


if __name__ == "__main__":
    main()

import sys
sys.path.insert(0, ".")
import numpy as np
from funygrad import Tensor


def test_creation():
    t = Tensor([1, 2, 3])
    assert np.allclose(t.numpy(), [1, 2, 3])
    assert t.shape == (3,)
    assert t.dtype == np.float32


def test_zeros():
    t = Tensor.zeros(2, 3)
    assert t.shape == (2, 3)
    assert np.allclose(t.numpy(), 0)


def test_ones():
    t = Tensor.ones(2, 3)
    assert np.allclose(t.numpy(), 1)


def test_randn():
    t = Tensor.randn(10, 10)
    assert t.shape == (10, 10)
    assert abs(t.numpy().mean()) < 0.5


def test_full():
    t = Tensor.full((3, 4), 7.0)
    assert t.shape == (3, 4)
    assert np.allclose(t.numpy(), 7.0)


def test_arange():
    t = Tensor.arange(0, 5)
    assert np.allclose(t.numpy(), [0, 1, 2, 3, 4])


def test_item():
    t = Tensor([3.14])
    assert abs(t.item() - 3.14) < 1e-6


def test_requires_grad():
    t = Tensor([1.0, 2.0], requires_grad=True)
    assert t.requires_grad
    assert t.grad is None


def test_detach():
    t = Tensor([1.0], requires_grad=True)
    d = t.detach()
    assert not d.requires_grad
    assert d.numpy() == t.numpy()


def test_T():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.T.shape == (2, 2)
    assert np.allclose(t.T.numpy(), [[1.0, 3.0], [2.0, 4.0]])


def test_chain_ops():
    x = Tensor.randn(5, 3)
    y = x.relu().exp().log()
    assert y.shape == (5, 3)


if __name__ == "__main__":
    for name, fn in list(locals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASSED {name}")
    print("\nAll tensor tests passed!")

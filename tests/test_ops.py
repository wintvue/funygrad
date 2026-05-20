import sys
sys.path.insert(0, ".")
import numpy as np
from funygrad import Tensor
from funygrad.lazy import LazyBuffer
from funygrad.ops import Ops


def test_add():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a + b
    assert np.allclose(c.numpy(), [5.0, 7.0, 9.0])


def test_sub():
    a = Tensor([4.0, 6.0, 8.0])
    b = Tensor([1.0, 2.0, 3.0])
    c = a - b
    assert np.allclose(c.numpy(), [3.0, 4.0, 5.0])


def test_mul():
    a = Tensor([2.0, 3.0, 4.0])
    b = Tensor([5.0, 6.0, 7.0])
    c = a * b
    assert np.allclose(c.numpy(), [10.0, 18.0, 28.0])


def test_div():
    a = Tensor([6.0, 8.0, 10.0])
    b = Tensor([2.0, 2.0, 2.0])
    c = a / b
    assert np.allclose(c.numpy(), [3.0, 4.0, 5.0])


def test_neg():
    a = Tensor([1.0, -2.0, 3.0])
    c = -a
    assert np.allclose(c.numpy(), [-1.0, 2.0, -3.0])


def test_matmul():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
    c = a @ b
    expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
    assert np.allclose(c.numpy(), expected)


def test_batch_matmul():
    a = Tensor(np.random.randn(4, 3, 5).astype(np.float32))
    b = Tensor(np.random.randn(5, 2).astype(np.float32))
    c = a @ b
    assert c.shape == (4, 3, 2)
    expected = a.numpy() @ b.numpy()
    assert np.allclose(c.numpy(), expected)


def test_relu():
    a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    c = a.relu()
    assert np.allclose(c.numpy(), [0.0, 0.0, 0.0, 1.0, 2.0])


def test_sigmoid():
    a = Tensor([0.0])
    c = a.sigmoid()
    assert np.allclose(c.numpy(), 0.5)


def test_tanh():
    a = Tensor([0.0])
    c = a.tanh()
    assert np.allclose(c.numpy(), 0.0)


def test_exp():
    a = Tensor([0.0, 1.0])
    c = a.exp()
    assert np.allclose(c.numpy(), [1.0, np.exp(1.0)])


def test_log():
    a = Tensor([1.0, np.exp(1.0)])
    c = a.log()
    assert np.allclose(c.numpy(), [0.0, 1.0])


def test_sum():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(a.sum().numpy(), 10.0)
    assert np.allclose(a.sum(axis=0).numpy(), [4.0, 6.0])
    assert np.allclose(a.sum(axis=1).numpy(), [3.0, 7.0])


def test_mean():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(a.mean().numpy(), 2.5)
    assert np.allclose(a.mean(axis=0).numpy(), [2.0, 3.0])


def test_max():
    a = Tensor([[1.0, 5.0], [3.0, 2.0]])
    assert np.allclose(a.max(axis=0).numpy(), [3.0, 5.0])
    assert np.allclose(a.max(axis=1).numpy(), [5.0, 3.0])


def test_reshape():
    a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = a.reshape(2, 3)
    assert b.shape == (2, 3)
    assert np.allclose(b.numpy(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_permute():
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = a.permute(1, 0)
    assert b.shape == (3, 2)
    assert np.allclose(b.numpy(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])


def test_expand():
    a = Tensor([1.0, 2.0, 3.0])
    b = a.expand(2, 3)
    assert b.shape == (2, 3)
    assert np.allclose(b.numpy(), [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])


def test_broadcasting():
    a = Tensor(np.ones((3, 4), dtype=np.float32))
    b = Tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    c = a + b
    assert c.shape == (3, 4)
    expected = np.ones((3, 4), dtype=np.float32) + np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(c.numpy(), expected)


def test_pow():
    a = Tensor([2.0, 3.0])
    b = Tensor([2.0, 3.0])
    c = a ** b
    assert np.allclose(c.numpy(), [4.0, 27.0])


def test_scalar_ops():
    a = Tensor([1.0, 2.0, 3.0])
    assert np.allclose((a + 2).numpy(), [3.0, 4.0, 5.0])
    assert np.allclose((a - 1).numpy(), [0.0, 1.0, 2.0])
    assert np.allclose((a * 3).numpy(), [3.0, 6.0, 9.0])
    assert np.allclose((a / 2).numpy(), [0.5, 1.0, 1.5])
    assert np.allclose((2 + a).numpy(), [3.0, 4.0, 5.0])
    assert np.allclose((2 - a).numpy(), [1.0, 0.0, -1.0])
    assert np.allclose((2 * a).numpy(), [2.0, 4.0, 6.0])
    assert np.allclose((6 / a).numpy(), [6.0, 3.0, 2.0])


if __name__ == "__main__":
    for name, fn in list(locals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASSED {name}")
    print("\nAll ops tests passed!")

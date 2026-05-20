import sys
sys.path.insert(0, ".")
import numpy as np
from funygrad import Tensor


def _numeric_grad(f, x, eps=1e-3):
    x64 = x.numpy().astype(np.float64)
    grad = np.zeros_like(x.numpy())
    for idx in np.ndindex(x.shape):
        orig = x64.copy()
        orig[idx] += eps
        fp = f(Tensor(orig.astype(np.float32))).numpy().sum()

        orig = x64.copy()
        orig[idx] -= eps
        fm = f(Tensor(orig.astype(np.float32))).numpy().sum()

        grad[idx] = (fp - fm) / (2 * eps)
    return grad


def _check_grad(f, x, tol=0.02):
    x_req = Tensor(x.numpy(), requires_grad=True)
    y = f(x_req)
    y.sum().backward()
    analytical = x_req.grad.numpy()
    numeric = _numeric_grad(f, x, eps=1e-3)
    err = np.max(np.abs(analytical - numeric))
    return err < tol, err


def test_add_grad():
    def f(x):
        return x + Tensor([1.0, 2.0, 3.0])
    ok, err = _check_grad(f, Tensor([4.0, 5.0, 6.0]))
    assert ok, f"grad error: {err}"


def test_mul_grad():
    def f(x):
        return x * Tensor([2.0, 3.0, 4.0])
    ok, err = _check_grad(f, Tensor([1.0, 2.0, 3.0]))
    assert ok, f"grad error: {err}"


def test_pow_grad():
    def f(x):
        return x ** 2
    ok, err = _check_grad(f, Tensor([1.0, 2.0, 3.0]))
    assert ok, f"grad error: {err}"


def test_div_grad():
    def f(x):
        return x / Tensor([2.0, 4.0, 8.0])
    ok, err = _check_grad(f, Tensor([3.0, 5.0, 7.0]))
    assert ok, f"grad error: {err}"


def test_neg_grad():
    def f(x):
        return -x
    ok, err = _check_grad(f, Tensor([1.0, -2.0, 3.0]))
    assert ok, f"grad error: {err}"


def test_exp_grad():
    def f(x):
        return x.exp()
    ok, err = _check_grad(f, Tensor([0.0, 1.0, 2.0]))
    assert ok, f"grad error: {err}"


def test_log_grad():
    def f(x):
        return x.log()
    ok, err = _check_grad(f, Tensor([1.0, 2.0, 3.0]))
    assert ok, f"grad error: {err}"


def test_relu_grad():
    def f(x):
        return x.relu()
    ok, err = _check_grad(f, Tensor([-1.0, 0.5, 2.0]))
    assert ok, f"grad error: {err}"


def test_sigmoid_grad():
    def f(x):
        return x.sigmoid()
    ok, err = _check_grad(f, Tensor([-1.0, 0.0, 1.0]))
    assert ok, f"grad error: {err}"


def test_tanh_grad():
    def f(x):
        return x.tanh()
    ok, err = _check_grad(f, Tensor([-1.0, 0.0, 1.0]))
    assert ok, f"grad error: {err}"


def test_sum_grad():
    def f(x):
        return x.sum()
    ok, err = _check_grad(f, Tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert ok, f"grad error: {err}"


def test_mean_grad():
    def f(x):
        return x.mean()
    ok, err = _check_grad(f, Tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert ok, f"grad error: {err}"


def test_matmul_grad():
    def f(a):
        b = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        return a @ b
    ok, err = _check_grad(f, Tensor(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)))
    assert ok, f"grad error: {err}"


def test_reshape_grad():
    def f(x):
        return x.reshape(2, 3)
    ok, err = _check_grad(f, Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert ok, f"grad error: {err}"


def test_permute_grad():
    def f(x):
        return x.permute(1, 0)
    ok, err = _check_grad(f, Tensor(np.random.randn(3, 4).astype(np.float32)))
    assert ok, f"grad error: {err}"


def test_expand_grad():
    def f(x):
        return x.expand(2, 3)
    ok, err = _check_grad(f, Tensor([1.0, 2.0, 3.0]))
    assert ok, f"grad error: {err}"


def test_broadcast_add_grad():
    def f(x):
        return x + Tensor(np.ones((3, 4), dtype=np.float32))
    ok, err = _check_grad(f, Tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)))
    assert ok, f"grad error: {err}"


def test_scalar_mul_grad():
    def f(x):
        return x * 3.0
    ok, err = _check_grad(f, Tensor([1.0, 2.0, 3.0]))
    assert ok, f"grad error: {err}"


def test_grad_accumulation():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y1 = x * 2
    y2 = x * 3
    z = y1.sum() + y2.sum()
    z.backward()
    assert np.allclose(x.grad.numpy(), [5.0, 5.0])


def test_chain_rule():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = (x * 2 + 1).relu()
    z = y.sum()
    z.backward()
    # dz/dx = dz/dy * dy/dx = 1 * (2 if x>0 else 0) * 2 = 2*sign
    assert np.allclose(x.grad.numpy(), [2.0, 2.0])


if __name__ == "__main__":
    for name, fn in list(locals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASSED {name}")
    print("\nAll autograd tests passed!")

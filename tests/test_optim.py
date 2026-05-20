import sys
sys.path.insert(0, ".")
import numpy as np
from funygrad import Tensor
from funygrad.optim import SGD


def test_sgd_basic():
    p = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    opt = SGD([p], lr=0.1)
    loss = (p * p).sum()
    loss.backward()
    opt.step()
    # dp/dx = 2x -> grad = [2, 4, 6]
    # new_x = x - lr * grad = [1 - 0.2, 2 - 0.4, 3 - 0.6] = [0.8, 1.6, 2.4]
    assert np.allclose(p.numpy(), [0.8, 1.6, 2.4])


def test_sgd_momentum():
    p = Tensor([2.0, 3.0], requires_grad=True)
    opt = SGD([p], lr=0.1, momentum=0.9)

    loss = (p * p).sum()
    loss.backward()
    opt.step()
    # grad = [4, 6], v = 0.9*0 + grad = [4, 6]
    # new_x = [2, 3] - 0.1 * [4, 6] = [1.6, 2.4]
    assert np.allclose(p.numpy(), [1.6, 2.4])

    p.zero_grad()
    loss2 = (p * p).sum()
    loss2.backward()
    opt.step()
    # grad = [3.2, 4.8], v = 0.9*[4,6] + [3.2,4.8] = [6.8, 10.2]
    # new_x = [1.6, 2.4] - 0.1 * [6.8, 10.2] = [0.92, 1.38]
    assert np.allclose(p.numpy(), [0.92, 1.38])


def test_sgd_weight_decay():
    p = Tensor([1.0, 1.0], requires_grad=True)
    opt = SGD([p], lr=0.1, weight_decay=0.5)

    loss = p.sum()
    loss.backward()
    opt.step()
    # grad = [1, 1] + 0.5 * [1, 1] = [1.5, 1.5]
    # new_x = [1, 1] - 0.1 * [1.5, 1.5] = [0.85, 0.85]
    assert np.allclose(p.numpy(), [0.85, 0.85])


def test_zero_grad():
    p = Tensor([1.0, 2.0], requires_grad=True)
    opt = SGD([p], lr=0.1)
    loss = (p * p).sum()
    loss.backward()
    opt.zero_grad()
    assert p.grad is None


def test_multiple_params():
    p1 = Tensor([1.0, 2.0], requires_grad=True)
    p2 = Tensor([3.0, 4.0], requires_grad=True)
    opt = SGD([p1, p2], lr=0.1)

    loss = p1.sum() + p2.sum()
    loss.backward()
    opt.step()

    assert np.allclose(p1.numpy(), [0.9, 1.9])  # [1,2] - 0.1*[1,1]
    assert np.allclose(p2.numpy(), [2.9, 3.9])  # [3,4] - 0.1*[1,1]


if __name__ == "__main__":
    for name, fn in list(locals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASSED {name}")
    print("\nAll optimizer tests passed!")

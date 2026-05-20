import sys
sys.path.insert(0, ".")
import numpy as np
from funygrad import Tensor, nn


def test_linear_shape():
    model = nn.Linear(10, 5)
    x = Tensor.randn(3, 10)
    out = model(x)
    assert out.shape == (3, 5)


def test_linear_no_bias():
    model = nn.Linear(10, 5, bias=False)
    x = Tensor.randn(3, 10)
    out = model(x)
    assert out.shape == (3, 5)
    assert model.bias is None


def test_linear_parameters():
    model = nn.Linear(10, 5)
    params = list(model.parameters())
    assert len(params) == 2
    assert params[0].shape == (5, 10)
    assert params[1].shape == (5,)


def test_relu():
    relu = nn.ReLU()
    x = Tensor([-1.0, 0.0, 2.0])
    out = relu(x)
    assert np.allclose(out.numpy(), [0.0, 0.0, 2.0])


def test_sequential():
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )
    x = Tensor.randn(3, 10)
    out = model(x)
    assert out.shape == (3, 2)
    assert len(list(model.parameters())) == 4


def test_mse_loss():
    criterion = nn.MSELoss()
    pred = Tensor([1.0, 2.0, 3.0])
    target = Tensor([1.0, 3.0, 5.0])
    loss = criterion(pred, target)
    expected = ((1-1)**2 + (2-3)**2 + (3-5)**2) / 3
    assert abs(loss.item() - expected) < 1e-6


def test_cross_entropy_shape():
    criterion = nn.CrossEntropyLoss()
    logits = Tensor.randn(4, 3)
    target = Tensor([0, 1, 2, 0])
    loss = criterion(logits, target)
    assert isinstance(loss.item(), float)


def test_state_dict():
    model = nn.Linear(3, 2)
    sd = model.state_dict()
    assert "weight" in sd
    assert sd["weight"].shape == (2, 3)
    assert "bias" in sd
    assert sd["bias"].shape == (2,)


def test_load_state_dict():
    model1 = nn.Linear(3, 2)
    model2 = nn.Linear(3, 2)
    sd = model1.state_dict()
    model2.load_state_dict(sd)
    assert np.allclose(model1.weight.numpy(), model2.weight.numpy())


def test_zero_grad():
    model = nn.Linear(3, 2)
    x = Tensor.randn(1, 3)
    out = model(x)
    loss = out.sum()
    loss.backward()
    model.zero_grad()
    for p in model.parameters():
        assert p.grad is None


def test_module_repr():
    model = nn.Linear(10, 5)
    assert "Linear" in repr(model)
    assert "parameters" in repr(model)


def test_train_eval():
    model = nn.Sequential(nn.Linear(3, 2), nn.ReLU())
    assert model._training
    model.eval()
    assert not model._training
    model.train()
    assert model._training


if __name__ == "__main__":
    for name, fn in list(locals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASSED {name}")
    print("\nAll nn tests passed!")

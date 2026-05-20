from __future__ import annotations
from typing import Optional, Tuple, Any, Union, List
import numpy as np

from funygrad.lazy import LazyBuffer
from funygrad.ops import Ops


class Tensor:
    __slots__ = ("lazydata", "requires_grad", "grad", "_ctx")

    def __init__(self, data: Union[np.ndarray, list, int, float, LazyBuffer, "Tensor"],
                 requires_grad: bool = False, device: str = "numpy", dtype: type = np.float32):
        if isinstance(data, LazyBuffer):
            self.lazydata = data
        elif isinstance(data, Tensor):
            self.lazydata = data.lazydata
            requires_grad = requires_grad or data.requires_grad
        else:
            arr = np.array(data, dtype=dtype)
            self.lazydata = LazyBuffer.from_numpy(arr, device)
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
        self._ctx: Optional[Function] = None

    def __repr__(self) -> str:
        arr = self.numpy()
        if arr.size == 1:
            return f"Tensor({arr.item():.6f}, requires_grad={self.requires_grad})"
        return f"Tensor({arr}, requires_grad={self.requires_grad})"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.lazydata.shape

    @property
    def dtype(self) -> type:
        return self.lazydata.dtype

    @property
    def device(self) -> str:
        return self.lazydata.device

    def numpy(self) -> np.ndarray:
        return self.lazydata.realize().copy()

    def item(self) -> float:
        arr = self.numpy()
        return arr.item()

    def detach(self) -> Tensor:
        return Tensor(self.lazydata, requires_grad=False)

    @staticmethod
    def zeros(*shape: int, requires_grad: bool = False, device: str = "numpy") -> Tensor:
        arr = np.zeros(shape, dtype=np.float32)
        return Tensor(arr, requires_grad=requires_grad, device=device)

    @staticmethod
    def ones(*shape: int, requires_grad: bool = False, device: str = "numpy") -> Tensor:
        arr = np.ones(shape, dtype=np.float32)
        return Tensor(arr, requires_grad=requires_grad, device=device)

    @staticmethod
    def randn(*shape: int, requires_grad: bool = False, device: str = "numpy") -> Tensor:
        arr = np.random.randn(*shape).astype(np.float32)
        return Tensor(arr, requires_grad=requires_grad, device=device)

    @staticmethod
    def uniform(*shape: int, low: float = 0.0, high: float = 1.0,
                requires_grad: bool = False, device: str = "numpy") -> Tensor:
        arr = np.random.uniform(low, high, shape).astype(np.float32)
        return Tensor(arr, requires_grad=requires_grad, device=device)

    @staticmethod
    def full(shape: tuple[int, ...], fill_value: float,
             requires_grad: bool = False, device: str = "numpy") -> Tensor:
        arr = np.full(shape, fill_value, dtype=np.float32)
        return Tensor(arr, requires_grad=requires_grad, device=device)

    @staticmethod
    def arange(start: int, end: int, step: int = 1,
               requires_grad: bool = False, device: str = "numpy") -> Tensor:
        arr = np.arange(start, end, step, dtype=np.float32)
        return Tensor(arr, requires_grad=requires_grad, device=device)

    def realize(self) -> np.ndarray:
        return self.lazydata.realize()

    def backward(self):
        assert self.shape == () or self.shape == (1,), f"backward requires scalar, got {self.shape}"
        self.grad = Tensor.ones(*self.shape)
        _backward(self)

    def zero_grad(self):
        self.grad = None

    def assign(self, other: Tensor) -> None:
        self.lazydata = other.lazydata

    def __add__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = _to_tensor(other)
        lb = self.lazydata.e(Ops.ADD, other.lazydata)
        return _maybe_autograd(Ops.ADD, lb, self, other)

    def __radd__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = _to_tensor(other)
        lb = self.lazydata.e(Ops.SUB, other.lazydata)
        return _maybe_autograd(Ops.SUB, lb, self, other)

    def __rsub__(self, other: Union[Tensor, float, int]) -> Tensor:
        return _to_tensor(other).__sub__(self)

    def __mul__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = _to_tensor(other)
        lb = self.lazydata.e(Ops.MUL, other.lazydata)
        return _maybe_autograd(Ops.MUL, lb, self, other)

    def __rmul__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = _to_tensor(other)
        lb = self.lazydata.e(Ops.DIV, other.lazydata)
        return _maybe_autograd(Ops.DIV, lb, self, other)

    def __rtruediv__(self, other: Union[Tensor, float, int]) -> Tensor:
        return _to_tensor(other).__truediv__(self)

    def __neg__(self) -> Tensor:
        lb = self.lazydata.e(Ops.NEG)
        return _maybe_autograd(Ops.NEG, lb, self)

    def __pow__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = _to_tensor(other)
        lb = self.lazydata.e(Ops.POW, other.lazydata)
        return _maybe_autograd(Ops.POW, lb, self, other)

    def __matmul__(self, other: Tensor) -> Tensor:
        lb = self.lazydata.e(Ops.MATMUL, other.lazydata)
        return _maybe_autograd(Ops.MATMUL, lb, self, other)

    def sum(self, axis: Optional[int | tuple[int, ...]] = None) -> Tensor:
        lb = self.lazydata.e(Ops.SUM, arg=axis)
        return _maybe_autograd(Ops.SUM, lb, self, axis=axis)

    def mean(self, axis: Optional[int | tuple[int, ...]] = None) -> Tensor:
        lb = self.lazydata.e(Ops.MEAN, arg=axis)
        return _maybe_autograd(Ops.MEAN, lb, self, axis=axis)

    def max(self, axis: Optional[int | tuple[int, ...]] = None) -> Tensor:
        lb = self.lazydata.e(Ops.MAX, arg=axis)
        return _maybe_autograd(Ops.MAX, lb, self, axis=axis)

    def relu(self) -> Tensor:
        lb = self.lazydata.e(Ops.RELU)
        return _maybe_autograd(Ops.RELU, lb, self)

    def sigmoid(self) -> Tensor:
        lb = self.lazydata.e(Ops.SIGMOID)
        return _maybe_autograd(Ops.SIGMOID, lb, self)

    def tanh(self) -> Tensor:
        lb = self.lazydata.e(Ops.TANH)
        return _maybe_autograd(Ops.TANH, lb, self)

    def exp(self) -> Tensor:
        lb = self.lazydata.e(Ops.EXP)
        return _maybe_autograd(Ops.EXP, lb, self)

    def log(self) -> Tensor:
        lb = self.lazydata.e(Ops.LOG)
        return _maybe_autograd(Ops.LOG, lb, self)

    def reshape(self, *shape: int) -> Tensor:
        lb = self.lazydata.e(Ops.RESHAPE, arg=shape)
        return _maybe_autograd(Ops.RESHAPE, lb, self, orig_shape=self.shape)

    def permute(self, *dims: int) -> Tensor:
        lb = self.lazydata.e(Ops.PERMUTE, arg=dims)
        return _maybe_autograd(Ops.PERMUTE, lb, self, dims=dims)

    def expand(self, *shape: int) -> Tensor:
        lb = self.lazydata.e(Ops.EXPAND, arg=shape)
        return _maybe_autograd(Ops.EXPAND, lb, self, orig_shape=self.shape)

    @property
    def T(self) -> Tensor:
        dims = tuple(range(len(self.shape) - 1, -1, -1))
        return self.permute(*dims)


def _to_tensor(x: Union[Tensor, float, int, np.ndarray, list]) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(x, requires_grad=False)


def _maybe_autograd(op: Ops, lb: LazyBuffer, *inputs: Tensor, **kwargs) -> Tensor:
    needs_grad = any(inp.requires_grad for inp in inputs)
    if not needs_grad:
        return Tensor(lb, requires_grad=False)

    ctx = Function(op=op, inputs=inputs, kwargs=kwargs)
    ctx.save_for_backward(*inputs)
    out = Tensor(lb, requires_grad=True)
    out._ctx = ctx
    ctx.output = out
    return out


def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if grad.shape == target_shape:
        return grad
    ndim_diff = len(grad.shape) - len(target_shape)
    padded_target = (1,) * ndim_diff + target_shape
    sum_axes = []
    for i, (gs, ts) in enumerate(zip(grad.shape, padded_target)):
        if ts == 1 and gs > 1:
            sum_axes.append(i)
    if sum_axes:
        grad = grad.sum(axis=tuple(sum_axes), keepdims=True)
    if ndim_diff > 0 and len(grad.shape) > len(target_shape):
        grad = grad.reshape(target_shape)
    return grad


class Function:
    def __init__(self, op: Ops = None, inputs: tuple[Tensor, ...] = (),
                 kwargs: dict = None):
        self.op = op
        self.inputs = inputs
        self.kwargs = kwargs or {}
        self.saved_tensors: List[Tensor] = []
        self.output: Optional[Tensor] = None

    def save_for_backward(self, *tensors: Tensor):
        self.saved_tensors.extend(tensors)


def _backward(root: Tensor):
    visited: set[int] = set()
    topo: list[Function] = []

    def build_topo(t: Tensor):
        if t._ctx is not None and id(t._ctx) not in visited:
            visited.add(id(t._ctx))
            for inp in t._ctx.inputs:
                if inp.requires_grad:
                    build_topo(inp)
            topo.append(t._ctx)

    build_topo(root)

    grad_map: dict[int, np.ndarray] = {id(root): root.grad.numpy()}

    for ctx in reversed(topo):
        grad_output = _get_grad(ctx, grad_map)
        if grad_output is None:
            continue

        grads = _backward_op(ctx, grad_output)
        if grads is None:
            continue

        for i, inp in enumerate(ctx.inputs):
            if not inp.requires_grad or grads[i] is None:
                continue
            _accumulate_grad(inp, grads[i], grad_map)


def _get_grad(ctx: Function, grad_map: dict[int, np.ndarray]) -> Optional[np.ndarray]:
    if ctx.output is not None:
        return grad_map.get(id(ctx.output))
    for inp in ctx.inputs:
        g = grad_map.get(id(inp))
        if g is not None:
            return g
    return None


def _backward_op(ctx: Function, grad_output: np.ndarray) -> Optional[List[Optional[np.ndarray]]]:
    op = ctx.op
    saved = ctx.saved_tensors
    kwargs = ctx.kwargs

    if op == Ops.ADD:
        a, b = saved[0], saved[1]
        return [_unbroadcast(grad_output, a.shape), _unbroadcast(grad_output, b.shape)]

    if op == Ops.SUB:
        a, b = saved[0], saved[1]
        return [_unbroadcast(grad_output, a.shape), _unbroadcast(-grad_output, b.shape)]

    if op == Ops.MUL:
        a, b = saved[0], saved[1]
        return [_unbroadcast(grad_output * b.numpy(), a.shape),
                _unbroadcast(grad_output * a.numpy(), b.shape)]

    if op == Ops.DIV:
        a, b = saved[0], saved[1]
        b_data = b.numpy()
        ga = _unbroadcast(grad_output / b_data, a.shape)
        gb = _unbroadcast(-grad_output * a.numpy() / (b_data * b_data), b.shape)
        return [ga, gb]

    if op == Ops.POW:
        a, b = saved[0], saved[1]
        a_data = a.numpy()
        b_data = b.numpy()
        pow_val = a_data ** (b_data - 1)
        ga = _unbroadcast(grad_output * b_data * pow_val, a.shape)
        gb_raw = grad_output * (a_data ** b_data) * np.log(np.maximum(a_data, 1e-10))
        if len(b.shape) == 0 or b.shape == ():
            gb = np.sum(gb_raw)
        else:
            gb = _unbroadcast(gb_raw, b.shape)
        return [ga, gb]

    if op == Ops.EXP:
        a = saved[0]
        return [grad_output * np.exp(a.numpy())]

    if op == Ops.LOG:
        a = saved[0]
        return [grad_output / np.maximum(a.numpy(), 1e-10)]

    if op == Ops.NEG:
        return [-grad_output]

    if op == Ops.RELU:
        a = saved[0]
        return [grad_output * (a.numpy() > 0)]

    if op == Ops.SIGMOID:
        a = saved[0]
        s = 1.0 / (1.0 + np.exp(-a.numpy()))
        return [grad_output * s * (1.0 - s)]

    if op == Ops.TANH:
        a = saved[0]
        t = np.tanh(a.numpy())
        return [grad_output * (1.0 - t * t)]

    if op == Ops.MATMUL:
        a, b = saved[0], saved[1]
        a_data = a.numpy()
        b_data = b.numpy()
        if len(a.shape) == 1 and len(b.shape) == 1:
            ga = grad_output * b_data
            gb = grad_output * a_data
        elif len(a.shape) == 1 and len(b.shape) == 2:
            ga = grad_output @ b_data.T
            gb = np.outer(a_data, grad_output)
        elif len(a.shape) == 2 and len(b.shape) == 1:
            ga = np.outer(grad_output, b_data)
            gb = a_data.T @ grad_output
        elif len(a.shape) == 2 and len(b.shape) == 2:
            ga = grad_output @ b_data.T
            gb = a_data.T @ grad_output
        elif len(a.shape) == 3 and len(b.shape) == 2:
            ga = grad_output @ b_data.T
            gb = np.sum(a_data.transpose(0, 2, 1) @ grad_output, axis=0)
        elif len(a.shape) == 2 and len(b.shape) == 3:
            gb = grad_output.transpose(0, 2, 1) @ a_data
            ga = np.sum(grad_output @ b_data.transpose(0, 2, 1), axis=0)
        else:
            ga = grad_output @ b_data.T
            gb = a_data.T @ grad_output
        return [ga, gb]

    if op == Ops.SUM:
        a = saved[0]
        axis = kwargs.get("axis", None)
        return [_expand_grad(grad_output, a.shape, axis)]

    if op == Ops.MEAN:
        a = saved[0]
        axis = kwargs.get("axis", None)
        if axis is None:
            n = int(np.prod(a.shape))
        elif isinstance(axis, int):
            ax = axis if axis >= 0 else axis + len(a.shape)
            n = a.shape[ax]
        else:
            n = 1
            for ax in axis:
                n *= a.shape[ax if ax >= 0 else ax + len(a.shape)]
        return [_expand_grad(grad_output, a.shape, axis) / n]

    if op == Ops.MAX:
        a = saved[0]
        axis = kwargs.get("axis", None)
        a_data = a.numpy()
        mask = (a_data == a_data.max(axis=axis, keepdims=True))
        expanded = _expand_grad(grad_output, a.shape, axis)
        return [expanded * mask]

    if op == Ops.RESHAPE:
        orig_shape = kwargs["orig_shape"]
        return [grad_output.reshape(orig_shape)]

    if op == Ops.PERMUTE:
        dims = kwargs["dims"]
        inv_dims = tuple(np.argsort(dims))
        return [np.transpose(grad_output, inv_dims)]

    if op == Ops.EXPAND:
        orig_shape = kwargs["orig_shape"]
        return [_unbroadcast(grad_output, orig_shape)]

    return None


def _expand_grad(grad: np.ndarray, target_shape: tuple[int, ...],
                 axis: Optional[int | tuple[int, ...]]) -> np.ndarray:
    if axis is None:
        return np.broadcast_to(grad.reshape((1,) * len(target_shape)), target_shape)
    if isinstance(axis, int):
        axis = (axis,)
    axis = tuple(a if a >= 0 else a + len(target_shape) for a in axis)
    shape = list(target_shape)
    for ax in sorted(axis, reverse=True):
        shape[ax] = 1
    return np.broadcast_to(grad.reshape(tuple(shape)), target_shape)


def _accumulate_grad(tensor: Tensor, grad: np.ndarray, grad_map: dict[int, np.ndarray]):
    grad = grad.astype(np.float32)
    if tensor.grad is None:
        tensor.grad = Tensor(grad, requires_grad=False)
        grad_map[id(tensor)] = grad
    else:
        existing = tensor.grad.numpy()
        new_grad = existing + grad
        tensor.grad = Tensor(new_grad, requires_grad=False)
        grad_map[id(tensor)] = new_grad

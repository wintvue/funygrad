from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np

from funygrad.ops import Ops, UNARY_OPS, BINARY_OPS, REDUCE_OPS, SHAPE_OPS


def _elementwise_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    if a == b:
        return a
    a_dims = (1,) * (len(b) - len(a)) + a
    b_dims = (1,) * (len(a) - len(b)) + b
    result = []
    for ad, bd in zip(a_dims, b_dims):
        if ad == bd:
            result.append(ad)
        elif ad == 1:
            result.append(bd)
        elif bd == 1:
            result.append(ad)
        else:
            raise ValueError(f"Cannot broadcast shapes {a} and {b}")
    return tuple(result)


def _reduce_shape(shape: tuple[int, ...], axis: Optional[int | tuple[int, ...]]) -> tuple[int, ...]:
    if axis is None:
        return ()
    if isinstance(axis, int):
        axis = (axis,)
    axis = tuple(a if a >= 0 else a + len(shape) for a in axis)
    return tuple(d for i, d in enumerate(shape) if i not in axis)


def _matmul_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    if len(a) == 1 and len(b) == 1:
        return ()
    if len(a) == 1 and len(b) == 2:
        return (b[1],)
    if len(a) == 2 and len(b) == 1:
        return (a[0],)
    if len(a) == 2 and len(b) == 2:
        return (a[0], b[1])
    if len(a) == 3 and len(b) == 2:
        return (a[0], a[1], b[1])
    if len(a) == 2 and len(b) == 3:
        return (b[0], a[0], b[2])
    raise ValueError(f"matmul shape mismatch: {a} @ {b}")


@dataclass
class LazyBuffer:
    op: Optional[Ops] = None
    srcs: tuple[LazyBuffer, ...] = field(default_factory=tuple)
    arg: Any = None
    shape: tuple[int, ...] = field(default_factory=tuple)
    dtype: type = np.float32
    device: str = "numpy"
    _realized: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.op is not None and not self.srcs:
            raise ValueError(f"LazyBuffer with op {self.op} must have sources")

    @classmethod
    def from_numpy(cls, arr: np.ndarray, device: str = "numpy") -> LazyBuffer:
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return cls(
            op=None,
            shape=arr.shape,
            dtype=np.float32,
            device=device,
            _realized=arr,
        )

    @classmethod
    def const(cls, shape: tuple[int, ...], val: float, device: str = "numpy") -> LazyBuffer:
        arr = np.full(shape, val, dtype=np.float32)
        return cls.from_numpy(arr, device)

    def realize(self) -> np.ndarray:
        if self._realized is not None:
            return self._realized
        from funygrad.codegen.numpy import execute
        result = execute(self)
        self._realized = result
        return result

    def is_realized(self) -> bool:
        return self._realized is not None

    @classmethod
    def _from_op(cls, op: Ops, srcs: tuple[LazyBuffer, ...], shape: tuple[int, ...],
                 arg: Any = None, dtype: type = np.float32, device: str = "numpy") -> LazyBuffer:
        return cls(op=op, srcs=srcs, arg=arg, shape=shape, dtype=dtype, device=device)

    def e(self, op: Ops, *srcs: LazyBuffer, arg: Any = None) -> LazyBuffer:
        """Element-wise or general binary op."""
        if op in BINARY_OPS and op != Ops.MATMUL:
            shape = _elementwise_shape(self.shape, srcs[0].shape)
            return LazyBuffer._from_op(op, (self, *srcs), shape, arg, self.dtype, self.device)
        if op in UNARY_OPS:
            return LazyBuffer._from_op(op, (self,), self.shape, arg, self.dtype, self.device)
        if op in REDUCE_OPS:
            shape = _reduce_shape(self.shape, arg)
            return LazyBuffer._from_op(op, (self,), shape, arg, self.dtype, self.device)
        if op == Ops.MATMUL:
            shape = _matmul_shape(self.shape, srcs[0].shape)
            return LazyBuffer._from_op(op, (self, *srcs), shape, arg, self.dtype, self.device)
        if op in SHAPE_OPS:
            if op == Ops.RESHAPE:
                new_shape = arg
            elif op == Ops.PERMUTE:
                new_shape = tuple(self.shape[i] for i in arg)
            elif op == Ops.EXPAND:
                new_shape = arg
            else:
                new_shape = self.shape
            return LazyBuffer._from_op(op, (self,), new_shape, arg, self.dtype, self.device)
        raise ValueError(f"Unknown op: {op}")

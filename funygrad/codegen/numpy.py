from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from funygrad.ops import Ops

if TYPE_CHECKING:
    from funygrad.lazy import LazyBuffer


def execute(lb: LazyBuffer) -> np.ndarray:
    if lb._realized is not None:
        return lb._realized

    srcs = [execute(s) for s in lb.srcs]
    op = lb.op
    arg = lb.arg

    if op == Ops.ADD:
        result = srcs[0] + srcs[1]
    elif op == Ops.SUB:
        result = srcs[0] - srcs[1]
    elif op == Ops.MUL:
        result = srcs[0] * srcs[1]
    elif op == Ops.DIV:
        result = srcs[0] / srcs[1]
    elif op == Ops.NEG:
        result = -srcs[0]
    elif op == Ops.POW:
        result = srcs[0] ** srcs[1]
    elif op == Ops.MATMUL:
        result = srcs[0] @ srcs[1]
    elif op == Ops.RELU:
        result = np.maximum(srcs[0], 0)
    elif op == Ops.SIGMOID:
        result = 1.0 / (1.0 + np.exp(-srcs[0]))
    elif op == Ops.TANH:
        result = np.tanh(srcs[0])
    elif op == Ops.EXP:
        result = np.exp(srcs[0])
    elif op == Ops.LOG:
        result = np.log(np.maximum(srcs[0], 1e-10))
    elif op == Ops.SUM:
        result = srcs[0].sum(axis=arg)
    elif op == Ops.MEAN:
        result = srcs[0].mean(axis=arg)
    elif op == Ops.MAX:
        result = srcs[0].max(axis=arg)
    elif op == Ops.RESHAPE:
        result = srcs[0].reshape(arg)
    elif op == Ops.PERMUTE:
        result = np.transpose(srcs[0], arg)
    elif op == Ops.EXPAND:
        result = np.broadcast_to(srcs[0], arg)
    elif op == Ops.MOVEMENT:
        result = srcs[0]
    else:
        raise ValueError(f"Unknown op: {op}")

    result = result.astype(np.float32)
    lb._realized = result
    return result

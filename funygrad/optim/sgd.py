from __future__ import annotations
from typing import List
import numpy as np
from funygrad.tensor import Tensor
from funygrad.nn.module import Parameter


class SGD:
    def __init__(self, params: List[Parameter], lr: float = 0.01,
                 momentum: float = 0.0, weight_decay: float = 0.0,
                 nesterov: bool = False):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self._velocity: dict[int, np.ndarray] = {}

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.numpy()

            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.numpy()

            if self.momentum != 0:
                pid = id(p)
                if pid not in self._velocity:
                    self._velocity[pid] = np.zeros_like(grad)
                self._velocity[pid] = self.momentum * self._velocity[pid] + grad

                if self.nesterov:
                    grad = grad + self.momentum * self._velocity[pid]
                else:
                    grad = self._velocity[pid]

            new_data = p.numpy() - self.lr * grad
            from funygrad.lazy import LazyBuffer
            p.lazydata = LazyBuffer.from_numpy(new_data.astype(np.float32))

from __future__ import annotations
from typing import Iterator, Any, Dict, List
import numpy as np

from funygrad.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data: np.ndarray | Tensor, requires_grad: bool = True):
        if isinstance(data, Tensor):
            super().__init__(data.lazydata, requires_grad=requires_grad)
        else:
            super().__init__(data, requires_grad=requires_grad)


class Module:
    def __init__(self):
        self._modules: Dict[str, Module] = {}
        self._params: Dict[str, Parameter] = {}
        self._training = True

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Parameter):
            self._params[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def parameters(self) -> Iterator[Parameter]:
        for p in self._params.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for name, p in self._params.items():
            yield f"{prefix}{name}", p
        for name, m in self._modules.items():
            yield from m.named_parameters(f"{prefix}{name}.")

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self) -> Dict[str, np.ndarray]:
        result = {}
        for name, p in self.named_parameters():
            result[name] = p.numpy()
        return result

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]):
        for name, p in self.named_parameters():
            if name in state_dict:
                arr = state_dict[name]
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                from funygrad.lazy import LazyBuffer
                p.lazydata = LazyBuffer.from_numpy(arr)

    def train(self, mode: bool = True):
        self._training = mode
        for m in self._modules.values():
            m.train(mode)

    def eval(self):
        self.train(False)

    def __repr__(self) -> str:
        params_count = sum(int(np.prod(p.shape)) for p in self.parameters())
        return f"{self.__class__.__name__}(parameters={params_count})"


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = list(layers)
        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

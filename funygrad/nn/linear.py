from __future__ import annotations
import math
from funygrad.tensor import Tensor
from funygrad.nn.module import Module, Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        bound = math.sqrt(1.0 / in_features)
        weight_arr = Tensor.uniform(out_features, in_features, low=-bound, high=bound)
        self.weight = Parameter(weight_arr)
        if bias:
            bias_arr = Tensor.uniform(out_features, low=-bound, high=bound)
            self.bias = Parameter(bias_arr)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

from __future__ import annotations
import numpy as np
from funygrad.tensor import Tensor
from funygrad.nn.module import Module


def _log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    insert_axis = axis if axis >= 0 else len(x.shape) + axis
    x_max = x.max(axis=axis)
    s1 = list(x_max.shape)
    s1.insert(insert_axis, 1)
    x_max = x_max.reshape(*s1)

    shifted = x - x_max
    log_sum_exp = shifted.exp().sum(axis=axis).log()
    s2 = list(log_sum_exp.shape)
    s2.insert(insert_axis, 1)
    log_sum_exp = log_sum_exp.reshape(*s2)
    return shifted - log_sum_exp


class CrossEntropyLoss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        n = logits.shape[0]
        num_classes = logits.shape[-1]

        target_np = target.numpy().flatten().astype(np.int64)
        mask = np.zeros((n, num_classes), dtype=np.float32)
        mask[np.arange(n), target_np] = 1.0
        mask_t = Tensor(mask, requires_grad=False)

        log_probs = _log_softmax(logits, axis=-1)

        losses = -(log_probs * mask_t)
        losses = losses.sum(axis=-1)

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        return losses


class MSELoss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        sq = diff * diff
        if self.reduction == "mean":
            return sq.mean()
        elif self.reduction == "sum":
            return sq.sum()
        return sq

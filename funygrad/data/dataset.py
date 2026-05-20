from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        ...


class TensorDataset(Dataset):
    def __init__(self, *tensors: np.ndarray):
        lengths = {len(t) for t in tensors}
        if len(lengths) > 1:
            raise ValueError(f"All tensors must have same first dimension, got {lengths}")
        self.tensors = [t.astype(np.float32) if t.dtype != np.float32 else t for t in tensors]

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, index: int) -> tuple[np.ndarray, ...]:
        return tuple(t[index] for t in self.tensors)

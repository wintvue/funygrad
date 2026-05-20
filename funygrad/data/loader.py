from __future__ import annotations
from typing import Iterator, Any
import numpy as np
from funygrad.data.dataset import Dataset


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 32,
                 shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[Any]:
        n = len(self.dataset)
        indices = list(range(n))
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n:
                if self.drop_last:
                    break
                end = n
            batch_indices = indices[start:end]
            yield self.dataset[batch_indices]

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

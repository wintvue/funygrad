from __future__ import annotations
from typing import List, Callable, Any


class Compose:
    def __init__(self, *transforms: Callable[[Any], Any]):
        self.transforms = list(transforms)

    def __call__(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)
        return x

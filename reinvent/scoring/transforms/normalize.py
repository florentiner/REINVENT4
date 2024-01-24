"""Normalize functions"""

__all__ = ["Normalize"]
from dataclasses import dataclass

import numpy as np

from .transform import Transform


@dataclass
class Parameters:
    type: str
    low: float
    high: float
    min_f:bool


def normalize(
    x: float, low: float, high: float, min_f:bool
):
    value = np.clip(x, low, high)
    score = (high - value)/(high-low)
    round_score = float(round(score, 3))
    return round_score if min_f else 1 - round_score


class Normalize(Transform, param_cls=Parameters):
    def __init__(self, params: Parameters):
        super().__init__(params)

        self.low = params.low
        self.high = params.high
        self.min_f = params.min_f

    def __call__(self, values) -> np.ndarray:
        transformed = [
            normalize(val, self.low, self.high, self.min_f)
            for val in values
        ]

        return np.array(transformed, dtype=np.float32)

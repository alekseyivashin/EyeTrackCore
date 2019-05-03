from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Fixation:
    start_time: int
    end_time: int
    duration: int
    end_x: float
    end_y: float


@dataclass
class FixationsGroup:
    size: int
    x: np.ndarray
    y: np.ndarray
    duration: np.ndarray

    @staticmethod
    def get(fixations: List[Fixation]) -> 'FixationsGroup':
        size = len(fixations)
        x = np.zeros(size)
        y = np.zeros(size)
        duration = np.zeros(size)
        for i, fixation in enumerate(fixations):
            x[i] = fixation.end_x
            y[i] = fixation.end_y
            duration[i] = fixation.duration
        return FixationsGroup(size, x, y, duration)

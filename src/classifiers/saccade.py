from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Saccade:
    start_time: int
    end_time: int
    duration: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float


@dataclass
class SaccadesGroup:
    size: int
    start_time: np.ndarray
    end_time: np.ndarray
    duration: np.ndarray
    start_x: np.ndarray
    start_y: np.ndarray
    end_x: np.ndarray
    end_y: np.ndarray

    @staticmethod
    def get(saccades: List[Saccade]) -> 'SaccadesGroup':
        size = len(saccades)
        start_time = np.zeros(size)
        end_time = np.zeros(size)
        duration = np.zeros(size)
        start_x = np.zeros(size)
        start_y = np.zeros(size)
        end_x = np.zeros(size)
        end_y = np.zeros(size)
        for i, saccade in enumerate(saccades):
            start_time[i] = saccade.start_time
            end_time[i] = saccade.end_time
            duration[i] = saccade.duration
            start_x[i] = saccade.start_x
            start_y[i] = saccade.start_y
            end_x[i] = saccade.end_x
            end_y[i] = saccade.end_y
        return SaccadesGroup(size, start_time, end_time, duration, start_x, start_y, end_x, end_y)

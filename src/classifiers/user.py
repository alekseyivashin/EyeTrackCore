from dataclasses import dataclass
from typing import List

import numpy as np

from json_parser.mapper import PositionInCoordinates


@dataclass
class UserPositionGroup:
    size: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    time_array: np.ndarray

    @staticmethod
    def get(positions: List[PositionInCoordinates], time_array: np.ndarray) -> 'UserPositionGroup':
        size = len(positions)
        x = np.zeros(size)
        y = np.zeros(size)
        z = np.zeros(size)
        for i, position in enumerate(positions):
            x[i] = position.x
            y[i] = position.y
            z[i] = position.z
        return UserPositionGroup(size, x, y, z, time_array)
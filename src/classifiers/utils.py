from typing import List, Tuple

import numpy as np

from json_parser.mapper import PositionOnDisplayArea


def positions_to_numpy_array(positions: List[PositionOnDisplayArea], display_size: Tuple[int, int]) \
        -> Tuple[np.ndarray, np.ndarray]:
    x_array = np.array(list(map(lambda position: position.x * display_size[0], positions)))
    y_array = np.array(list(map(lambda position: position.y * display_size[1], positions)))
    return x_array, y_array


def time_to_numpy_array(times: List[int]) -> np.ndarray:
    return (np.array(times) / 1000).astype(int)

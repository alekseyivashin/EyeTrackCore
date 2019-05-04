from typing import Tuple

import numpy as np

from classifiers.fixation import FixationsGroup
from classifiers.saccade import SaccadesGroup
from learn.vector import Vector


def get_features_vector(fixations: FixationsGroup, saccades: SaccadesGroup) -> Vector:
    return Vector(get_count_of_fixations(fixations),
                  get_average_duration_of_fixations(fixations),
                  get_average_horizontal_amplitude_of_saccades(saccades),
                  get_average_velocity_of_saccades(saccades))


def get_count_of_fixations(fixations: FixationsGroup) -> int:
    return fixations.size


def get_average_duration_of_fixations(fixations: FixationsGroup) -> float:
    return np.average(fixations.duration)


def get_average_horizontal_amplitude_of_saccades(saccades: SaccadesGroup) -> float:
    tg05 = np.tan(np.deg2rad(0.5))
    horizontal_amplitude = saccades.end_x - saccades.start_x
    vertical_amplitude = saccades.end_y - saccades.start_y
    return np.average([hor for i, hor in enumerate(horizontal_amplitude) if hor / vertical_amplitude[i] > tg05])


def get_average_velocity_of_saccades(saccades: SaccadesGroup) -> float:
    horizontal_amplitude = saccades.end_x - saccades.start_x
    vertical_amplitude = saccades.end_y - saccades.start_y
    return np.average((horizontal_amplitude ** 2 + vertical_amplitude ** 2) ** 0.5 / saccades.duration)

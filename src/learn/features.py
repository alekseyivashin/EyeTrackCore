from typing import Tuple, List

import numpy as np
from scipy.spatial import distance

from classifiers.fixation import FixationsGroup
from classifiers.saccade import SaccadesGroup
from classifiers.user import UserPositionGroup
from learn.vector import Vector


def get_features_vector(fixations: FixationsGroup, saccades: SaccadesGroup,
                        user_positions: UserPositionGroup) -> Vector:
    user_x, user_y, user_z = get_average_user_position(user_positions)
    return Vector(get_count_of_fixations(fixations),
                  get_average_duration_of_fixations(fixations),
                  get_average_horizontal_amplitude_of_saccades(saccades),
                  get_average_velocity_of_saccades(saccades),
                  np.abs(user_x),
                  np.abs(user_y),
                  np.abs(user_z),
                  get_average_user_velocity(user_positions))


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


def get_average_user_position(user_positions: UserPositionGroup) -> Tuple[float, float, float]:
    return np.average(user_positions.x), np.average(user_positions.y), np.average(user_positions.z)


def get_average_user_velocity(user_positions: UserPositionGroup) -> float:
    overall_distance = 0.0
    for i in range(user_positions.size - 1):
        overall_distance += distance.euclidean([user_positions.x[i], user_positions.y[i], user_positions.z[i]],
                                               [user_positions.x[i + 1], user_positions.y[i + 1],
                                                user_positions.z[i + 1]])
    return overall_distance / (user_positions.time_array.max() - user_positions.time_array.min())

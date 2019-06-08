from dataclasses import dataclass
from typing import List

from numpy import ndarray, array


@dataclass
class Vector:
    fixations_count: int
    fixations_average_duration: float
    saccades_average_horizontal_amplitude: float
    saccades_average_velocity: float
    # average_user_position_x: float
    # average_user_position_y: float
    # average_user_position_z: float
    # average_user_velocity: float

    def to_array(self) -> ndarray:
        return array([self.fixations_count,
                      self.fixations_average_duration,
                      self.saccades_average_horizontal_amplitude,
                      self.saccades_average_velocity,
                      # self.average_user_position_x,
                      # self.average_user_position_y,
                      # self.average_user_position_z,
                      # self.average_user_velocity
                      ])

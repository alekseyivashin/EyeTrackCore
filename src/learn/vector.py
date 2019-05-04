from dataclasses import dataclass

from numpy import ndarray, array


@dataclass
class Vector:
    fixations_count: int
    fixations_average_duration: float
    saccades_average_horizontal_amplitude: float
    saccades_average_velocity: float

    def to_array(self) -> ndarray:
        return array([self.fixations_count,
                      self.fixations_average_duration,
                      self.saccades_average_horizontal_amplitude,
                      self.saccades_average_velocity])

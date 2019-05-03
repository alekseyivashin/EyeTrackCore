from dataclasses import dataclass


@dataclass
class Saccade:
    start_time: int
    end_time: int
    duration: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float

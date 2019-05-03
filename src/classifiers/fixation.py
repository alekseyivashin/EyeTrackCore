from dataclasses import dataclass


@dataclass
class Fixation:
    start_time: int
    end_time: int
    duration: int
    end_x: float
    end_y: float

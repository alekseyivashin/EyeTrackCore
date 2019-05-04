from typing import List

from numpy import ndarray, array

from learn.vector import Vector


def vector_list_to_array(vectors: List[Vector]) -> ndarray:
    return array([vector.to_array() for vector in vectors])
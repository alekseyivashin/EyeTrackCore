from typing import List, Tuple

import json_parser.parser as parser
from classifiers.draw import draw_fixations
from classifiers.draw import draw_scanpath
from classifiers.fixation import FixationsGroup
from classifiers.saccade import SaccadesGroup
from json_parser.mapper import gaze_data_from_dict, GazeData
from classifiers.utils import positions_to_numpy_array
from classifiers.utils import time_to_numpy_array
from classifiers.detectors import fixation_detection
from classifiers.detectors import saccade_detection

import numpy as np
import matplotlib.pyplot as plt

from learn.features import get_features_vector
from learn.knn import KNN
from learn.vector import Vector

DISPLAY_SIZE = 1920, 1080


def main():
    lesha_train_vector, lesha_test_vector = get_vector_for_data("Алексей63688521416324.json")
    masha_train_vector, masha_test_vector = get_vector_for_data("Мария63688521639345.json")
    knn = KNN()
    knn.learn([lesha_train_vector, masha_train_vector], [lesha_test_vector, masha_test_vector], ["Lesha", "Masha"])
    a = 1
    # plot_fixations = draw_fixations(fixations, display_size)
    # plot_scanpath = draw_scanpath(fixations, saccades, display_size)
    # plot_fixations.show()
    # plot_scanpath.show()


def get_vector_for_data(filename: str) -> Tuple[Vector, Vector]:
    data = parser.parse(filename)
    gaze_data_list = gaze_data_from_dict(data)
    learn_data_list = gaze_data_list[ : int(len(gaze_data_list) * .5)]
    train_data_list = gaze_data_list[int(len(gaze_data_list) * .5) : ]
    return processing_data(learn_data_list), processing_data(train_data_list)

def processing_data(gaze_data_list: List[GazeData]) -> Vector:
    x_array, y_array = positions_to_numpy_array(list(map(lambda data: data.average_display_coordinate, gaze_data_list)),
                                                DISPLAY_SIZE)
    time_array = time_to_numpy_array(list(map(lambda data: data.system_time_stamp, gaze_data_list)))
    fixations = FixationsGroup.get(fixation_detection(x_array, y_array, time_array))
    saccades = SaccadesGroup.get(saccade_detection(x_array, y_array, time_array))
    return get_features_vector(fixations, saccades)


if __name__ == '__main__':
    main()

from typing import List, Dict

import json_parser.parser as parser
from classifiers.detectors import fixation_detection
from classifiers.detectors import saccade_detection
from classifiers.fixation import FixationsGroup
from classifiers.saccade import SaccadesGroup
from classifiers.utils import positions_to_numpy_array
from classifiers.utils import time_to_numpy_array
from json_parser.mapper import GazeData
from learn.features import get_features_vector
from learn.kmeans import KMeansMethod
from learn.knn import KNN
from learn.random_forest import RandomForest
from learn.svcmethod import SVCMethod
from learn.utils import LearnUtils
from learn.vector import Vector



DISPLAY_SIZE = 1920, 1080


def main():
    data = parser.parse_all()
    vectors = get_vectors_for_data(data)
    LearnUtils.set_up(vectors)

    #----------------------CLASSIFICATION----------------------#
    knn_result = KNN().learn()

    svc_result = SVCMethod().learn()

    random_forest_result = RandomForest().learn()

    # ----------------------CLUSTERING----------------------#
    kmeans_result = KMeansMethod().learn()
    a = 1
    # plot_fixations = draw_fixations(fixations, display_size)
    # plot_scanpath = draw_scanpath(fixations, saccades, display_size)
    # plot_fixations.show()
    # plot_scanpath.show()


def get_vectors_for_data(data: Dict[str, List[List[GazeData]]]) -> Dict[str, List[Vector]]:
    return {name: [processing_data(list) for list in data[name]] for name in data}

def processing_data(gaze_data_list: List[GazeData]) -> Vector:
    x_array, y_array = positions_to_numpy_array(list(map(lambda data: data.average_display_coordinate, gaze_data_list)),
                                                DISPLAY_SIZE)
    time_array = time_to_numpy_array(list(map(lambda data: data.system_time_stamp, gaze_data_list)))
    fixations = FixationsGroup.get(fixation_detection(x_array, y_array, time_array))
    saccades = SaccadesGroup.get(saccade_detection(x_array, y_array, time_array))
    return get_features_vector(fixations, saccades)


if __name__ == '__main__':
    main()

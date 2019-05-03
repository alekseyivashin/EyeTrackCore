import json_parser.parser as parser
from classifiers.draw import draw_fixations
from classifiers.draw import draw_scanpath
from json_parser.mapper import gaze_data_from_dict
from classifiers.utils import positions_to_numpy_array
from classifiers.utils import time_to_numpy_array
from classifiers.detectors import fixation_detection
from classifiers.detectors import saccade_detection

import matplotlib.pyplot as plt

def main():
    display_size = 1920, 1080
    data = parser.parse()
    gaze_data_list = gaze_data_from_dict(data)
    x_array, y_array = positions_to_numpy_array(list(map(lambda data: data.average_display_coordinate, gaze_data_list)), display_size)
    time_array = time_to_numpy_array(list(map(lambda data: data.system_time_stamp, gaze_data_list)))
    fixations = fixation_detection(x_array, y_array, time_array)
    saccades = saccade_detection(x_array, y_array, time_array)
    plot_fixations = draw_fixations(fixations, display_size)
    plot_scanpath = draw_scanpath(fixations, saccades, display_size)
    plot_fixations.show()
    plot_scanpath.show()
    a = 1

if __name__ == '__main__':
    main()
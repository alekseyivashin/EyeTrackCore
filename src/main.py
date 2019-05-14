from typing import List, Dict

import numpy as np
from sklearn.metrics import accuracy_score

import json_parser.parser as parser
from classifiers.detectors import fixation_detection
from classifiers.detectors import saccade_detection
from classifiers.fixation import FixationsGroup
from classifiers.saccade import SaccadesGroup
from classifiers.utils import positions_to_numpy_array
from classifiers.utils import time_to_numpy_array
from json_parser.mapper import GazeData
from learn.bayes import Bayes
from learn.features import get_features_vector
from learn.gaussian import Gaussian
from learn.knn import KNN
from learn.random_forest import RandomForest
from learn.svcmethod import SVCMethod
from learn.utils import LearnUtils
from learn.vector import Vector
from learn.voting import Voting
from report.plots import plot_confusion_matrix
from report.reports import print_classification_report

DISPLAY_SIZE = 1920, 1080


def main():
    print("Process started")
    data = parser.parse_all()
    print("Reading and parsing completed")
    vectors = get_vectors_for_data(data)
    print("Vectors created")
    LearnUtils.set_up(vectors, test_indexes=[3, 4])
    print("Utils setup completed")
    encoded_labels = np.repeat(LearnUtils.get_encoded_labels(), 2)
    # voting_result = Voting().learn()
    # print("Voting classifier learned")
    # voting_score = accuracy_score(encoded_labels, voting_result)

    # ----------------------CLASSIFICATION----------------------#
    # knn = KNN()
    # knn_result = knn.learn()
    # knn_score = accuracy_score(encoded_labels, knn_result)
    #
    # svc = SVCMethod()
    # svc_result = svc.learn()
    # svc_score = accuracy_score(encoded_labels, svc_result)
    #
    # random_forest = RandomForest()
    # random_forest_result = random_forest.learn()
    # random_forest_score = accuracy_score(encoded_labels, random_forest_result)
    #
    # gaussian = Gaussian()
    # gaussian_result = gaussian.learn()
    # gaussian_score = accuracy_score(encoded_labels, gaussian_result)
    #
    # bayes = Bayes()
    # bayes_result = bayes.learn()
    # bayes_score = accuracy_score(encoded_labels, bayes_result)

    # print_classification_report(encoded_labels, voting_result, LearnUtils.get_labels())
    # print(voting_score)
    #
    # # ----------------------CONFUSION MATRIX----------------------#
    # fig = plot_confusion_matrix(encoded_labels, voting_result, normalize=True,
    #                             title=f'Матрица смещения для обобщенных методов\nТочность оценки: {voting_score:.2f}')
    # fig.show()
    # fig1 = plot_confusion_matrix(encoded_labels, knn_result, normalize=True,
    #                              title=f'Матрица смещения для метода "K ближайших соседей"\nТочность оценки: {knn_score:.2f}')
    # fig2 = plot_confusion_matrix(encoded_labels, svc_result, normalize=True,
    #                              title=f'Матрица смещения для метода опорных векторов\nТочность оценки: {svc_score:.2f}')
    # fig3 = plot_confusion_matrix(encoded_labels, random_forest_result, normalize=True,
    #                              title=f'Матрица смещения для метода случайного леса\nТочность оценки: {random_forest_score:.2f}')
    # fig4 = plot_confusion_matrix(encoded_labels, gaussian_result, normalize=True,
    #                              title=f'Матрица смещения для метода преобразования Гаусса\nТочность оценки: {gaussian_score:.2f}')
    # fig5 = plot_confusion_matrix(encoded_labels, bayes_result, normalize=True,
    #                              title=f'Матрица смещения для метода найвного Байеса\nТочность оценки: {bayes_score:.2f}')
    # fig1.show()
    # fig2.show()
    # fig3.show()
    # fig4.show()
    # fig5.show()

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

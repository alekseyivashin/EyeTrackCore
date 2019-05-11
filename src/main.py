from typing import List, Dict

import json_parser.parser as parser
import numpy as np
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
from learn.vector import Vector
from sklearn.metrics import accuracy_score

from learn.utils import LearnUtils
from learn.voting import Voting
from report.plots import plot_confusion_matrix

DISPLAY_SIZE = 1920, 1080


def main():
    data = parser.parse_all()
    vectors = get_vectors_for_data(data)
    # knn_scores = []
    # svc_scores = []
    # random_forest_scores = []
    # gaussian_scores = []
    # bayes_scores = []
    # major_scores = []
    # for i in range(6):
    #     LearnUtils.set_up(vectors, test_index=i)
    #     encoded_labels = LearnUtils.get_encoded_labels()
    #
    #     knn_result = KNN().learn()
    #
    #     svc_result = SVCMethod().learn()
    #
    #     random_forest_result = RandomForest().learn()
    #
    #     gaussian_result = Gaussian().learn()
    #
    #     bayes_result = Bayes().learn()
    #
    #     major_result = get_major_result(encoded_labels,
    #                                     [knn_result, svc_result, random_forest_result, gaussian_result, bayes_result])
    #
    #     knn_scores.append(accuracy_score(encoded_labels, knn_result))
    #     svc_scores.append(accuracy_score(encoded_labels, svc_result))
    #     random_forest_scores.append(accuracy_score(encoded_labels, random_forest_result))
    #     gaussian_scores.append(accuracy_score(encoded_labels, gaussian_result))
    #     bayes_scores.append(accuracy_score(encoded_labels, bayes_result))
    #     major_scores.append(accuracy_score(encoded_labels, major_result))

    LearnUtils.set_up(vectors, test_indexes=[4])
    encoded_labels = np.repeat(LearnUtils.get_encoded_labels(), 1)
    voting_result = Voting().learn()
    voting_score = accuracy_score(encoded_labels, voting_result)

    # ----------------------CLASSIFICATION----------------------#
    knn_result = KNN().learn()
    knn_score = accuracy_score(encoded_labels, knn_result)

    svc_result = SVCMethod().learn()
    svc_score = accuracy_score(encoded_labels, svc_result)

    random_forest_result = RandomForest().learn()
    random_forest_score = accuracy_score(encoded_labels, random_forest_result)

    gaussian_result = Gaussian().learn()
    gaussian_score = accuracy_score(encoded_labels, gaussian_result)

    bayes_result = Bayes().learn()
    bayes_score = accuracy_score(encoded_labels, bayes_result)

    # print_classification_report(encoded_labels, major_result, LearnUtils.get_labels())

    fig = plot_confusion_matrix(encoded_labels, voting_result, normalize=True,
                                title=f'Матрица смещения для обобщенных методов без применения весов\nТочность оценки: {voting_score:.2f}')
    fig.show()
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

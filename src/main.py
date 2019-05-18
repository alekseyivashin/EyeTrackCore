import warnings

from learn.decision_tree import DecisionTree

warnings.filterwarnings("ignore", category=FutureWarning)

from typing import List, Dict

import numpy as np
from sklearn.metrics import accuracy_score

import json_parser.parser as parser
from classifiers.detectors import fixation_detection
from classifiers.detectors import saccade_detection
from classifiers.fixation import FixationsGroup
from classifiers.saccade import SaccadesGroup
from classifiers.user import UserPositionGroup
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
    LearnUtils.set_up(vectors, test_indexes=[1, 4])
    print("Utils setup completed")
    encoded_labels = np.repeat(LearnUtils.get_encoded_labels(), 2)
    times = 100
    knn = KNN()
    svc = SVCMethod()
    random_forest = RandomForest()
    gaussian = Gaussian()
    decision_tree = DecisionTree()
    bayes = Bayes()
    voting = Voting()
    # ----------------------VOTING LEARN----------------------#
    # voting_result = run_multiple_times(voting, encoded_labels, times)
    # a = 1

    # ----------------------VOTING FEATURE SELECTION----------#
    # scores = {}
    # scores["without"] = accuracy_score(encoded_labels, voting.learn())
    # for feature_method_name in voting.get_feature_method_names():
    #     print(f"Start {feature_method_name}")
    #     voting_result = voting.learn(feature_method_name)
    #     scores[feature_method_name] = accuracy_score(encoded_labels, voting_result)
    # a = 1

    # ----------------------VOTING CROSS VALIDATION-----------#
    # cross_val_result = voting.cross_validation()
    # a = 1

    # ----------------------GRID------------------------------#
    # knn_grid_result = knn.grid()
    # svc_grid_result = svc.grid()
    # random_forest_grid_result = random_forest.grid()
    # gaussian_grid_result = gaussian.grid()
    # decision_tree_grid_result = decision_tree.grid()
    # a = 1

    # ----------------------CLASSIFICATION--------------------#
    # knn_result = knn.learn()
    # knn_score = accuracy_score(encoded_labels, knn_result)
    #
    # svc_result = svc.learn()
    # svc_score = accuracy_score(encoded_labels, svc_result)
    #
    # random_forest_result = random_forest.learn()
    # random_forest_score = accuracy_score(encoded_labels, random_forest_result)
    #
    # gaussian_result = gaussian.learn()
    # gaussian_score = accuracy_score(encoded_labels, gaussian_result)
    #
    # decision_tree_result = decision_tree.learn()
    # decision_tree_score = accuracy_score(encoded_labels, decision_tree_result)
    #
    # bayes_result = bayes.learn()
    # bayes_score = accuracy_score(encoded_labels, bayes_result)
    # a = 1

    # ----------------------CLASSIFICATION MULTIPLE TIMES-----#
    # knn_score = run_multiple_times(knn, encoded_labels, times)
    svc_score = run_multiple_times(svc, encoded_labels, times)
    # random_forest_score = run_multiple_times(random_forest, encoded_labels, times)
    # gaussian_score = run_multiple_times(gaussian, encoded_labels, times)
    # decision_tree_score = run_multiple_times(decision_tree, encoded_labels, times)
    # bayes_score = run_multiple_times(bayes, encoded_labels, times)
    a = 1

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

def run_multiple_times(method, encoded_labels, times):
    score = []
    for i in range(times):
        print(f"Learn {method.__class__.__name__} number: {i + 1}")
        voting_result = method.learn()
        voting_score = accuracy_score(encoded_labels, voting_result)
        # print_classification_report(encoded_labels, voting_result, LearnUtils.get_labels())
        score.append(voting_score)
    return np.mean(score), np.std(score)


def get_vectors_for_data(data: Dict[str, List[List[GazeData]]]) -> Dict[str, List[Vector]]:
    return {name: [processing_data(list) for list in data[name]] for name in data}


def processing_data(gaze_data_list: List[GazeData]) -> Vector:
    x_array, y_array = positions_to_numpy_array(list(map(lambda data: data.average_display_coordinate, gaze_data_list)),
                                                DISPLAY_SIZE)
    time_array = time_to_numpy_array(list(map(lambda data: data.system_time_stamp, gaze_data_list)))
    fixations = FixationsGroup.get(fixation_detection(x_array, y_array, time_array))
    saccades = SaccadesGroup.get(saccade_detection(x_array, y_array, time_array))
    user_positions = UserPositionGroup.get(list(map(lambda data: data.average_user_coordinate, gaze_data_list)),
                                           time_array)
    return get_features_vector(fixations, saccades, user_positions)


if __name__ == '__main__':
    main()

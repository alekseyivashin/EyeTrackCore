from typing import List

import numpy as np
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from learn.utils import LearnUtils


class SVCMethod:

    def learn(self):
        labels, train_array, test_array = LearnUtils.get_learn_data()

        clf = self.get_classifier()
        clf.fit(train_array, labels)

        return clf.predict(test_array).tolist()

    def grid(self):
        train_labels, train_array, test_array = LearnUtils.get_learn_data()
        test_labels = np.repeat(LearnUtils.get_encoded_labels(), 2)
        ps = PredefinedSplit(np.append(np.full((train_array.shape[0]), -1, dtype=int),
                                       np.full((test_array.shape[0]), 0, dtype=int)))
        param_grid = dict(C=[0.001, 0.01, 0.1, 1, 10, 20, 100, 1000],
                          gamma=[0.001, 0.01, 0.1, 1, 2, 5, 10],
                          kernel=["linear", "poly", "rbf", "sigmoid"])
        clf = self.get_classifier()
        grid = GridSearchCV(clf, param_grid, cv=ps)
        grid.fit(np.append(train_array, test_array, axis=0), np.append(train_labels, test_labels, axis=0))
        return grid

    def get_classifier(self):
        return SVC(kernel="rbf", C=10, gamma=1, probability=True)
        # return SVC(probability=True)

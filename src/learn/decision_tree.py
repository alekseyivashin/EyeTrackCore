from typing import List

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from learn.utils import LearnUtils


class DecisionTree:

    def learn(self) -> List[int]:
        labels, train_array, test_array = LearnUtils.get_learn_data()

        clf = self.get_classifier()
        clf.fit(train_array, labels)

        return clf.predict(test_array).tolist()

    def grid(self):
        train_labels, train_array, test_array = LearnUtils.get_learn_data()
        test_labels = np.repeat(LearnUtils.get_encoded_labels(), 2)
        ps = PredefinedSplit(np.append(np.full((train_array.shape[0]), -1, dtype=int),
                                       np.full((test_array.shape[0]), 0, dtype=int)))
        param_grid = dict(
            criterion = ["gini", "entropy"],
            splitter = ["best", "random"],
            max_depth = [None, 1, 2, 3, 5, 10],
            min_samples_split  = [2, 3, 4, 5, 10],
            min_samples_leaf  = [1, 2, 3, 4, 5, 10],
        )
        clf = self.get_classifier()
        grid = GridSearchCV(clf, param_grid, cv=ps)
        grid.fit(np.append(train_array, test_array, axis=0), np.append(train_labels, test_labels, axis=0))
        return grid

    def get_classifier(self):
        return DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_leaf=1, min_samples_split=2, splitter="best")
        # return DecisionTreeClassifier()
from typing import List

import numpy as np
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from learn.utils import LearnUtils


class KNN:
    n_neighbors = 5

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
        param_grid = dict(n_neighbors=list(range(1, train_array.shape[0] - 1)))
        clf = self.get_classifier()
        grid = GridSearchCV(clf, param_grid, cv=ps)
        grid.fit(np.append(train_array, test_array, axis=0), np.append(train_labels, test_labels, axis=0))
        return grid

    def get_classifier(self):
        return KNeighborsClassifier(self.n_neighbors)

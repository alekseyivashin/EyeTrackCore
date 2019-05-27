from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit, GridSearchCV

from learn.utils import LearnUtils


class RandomForest:

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
            n_estimators=[10, 50, 100, 200, 500],
            max_features=["auto", "sqrt", "log2"],
            max_depth=[None, 2, 3, 4, 5, 6, 7, 8],
            criterion=["gini", "entropy"]
        )
        clf = self.get_classifier()
        grid = GridSearchCV(clf, param_grid, cv=ps)
        grid.fit(np.append(train_array, test_array, axis=0), np.append(train_labels, test_labels, axis=0))
        return grid

    def get_classifier(self):
        # return RandomForestClassifier(criterion="entropy", max_depth=4, max_features="sqrt", n_estimators=200)
        return RandomForestClassifier()

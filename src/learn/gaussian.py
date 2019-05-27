from typing import List

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import CompoundKernel, ConstantKernel, DotProduct, ExpSineSquared, Exponentiation, \
    Matern, PairwiseKernel, RationalQuadratic, WhiteKernel
from sklearn.model_selection import PredefinedSplit, GridSearchCV

from learn.utils import LearnUtils


class Gaussian:

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
            kernel=[ConstantKernel(), DotProduct(), Matern(), PairwiseKernel(), RationalQuadratic(), WhiteKernel()],
            n_restarts_optimizer = [0, 1, 2, 3],
            max_iter_predict = [10, 50, 100, 200]
        )
        clf = self.get_classifier()
        grid = GridSearchCV(clf, param_grid, cv=ps)
        grid.fit(np.append(train_array, test_array, axis=0), np.append(train_labels, test_labels, axis=0))
        return grid

    def get_classifier(self):
        # return GaussianProcessClassifier(kernel=DotProduct(), max_iter_predict=10, n_restarts_optimizer=0)
        return GaussianProcessClassifier()

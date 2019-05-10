from typing import List

from sklearn.gaussian_process import GaussianProcessClassifier

from learn.utils import LearnUtils


class Gaussian:

    def learn(self) -> List[int]:
        labels, train_array, test_array = LearnUtils.get_learn_data()

        clf = GaussianProcessClassifier()
        clf.fit(train_array, labels)

        return clf.predict(test_array).tolist()

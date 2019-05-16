from typing import List

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier

from learn.utils import LearnUtils


class DecisionTree:

    def learn(self) -> List[int]:
        labels, train_array, test_array = LearnUtils.get_learn_data()

        clf = DecisionTreeClassifier()
        clf.fit(train_array, labels)

        return clf.predict(test_array).tolist()

    def get_classifier(self):
        return DecisionTreeClassifier()
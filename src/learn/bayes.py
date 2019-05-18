from typing import List

from sklearn.naive_bayes import GaussianNB

from learn.utils import LearnUtils


class Bayes:

    def learn(self) -> List[int]:
        labels, train_array, test_array = LearnUtils.get_learn_data()

        clf = self.get_classifier()
        clf.fit(train_array, labels)

        return clf.predict(test_array).tolist()

    def get_classifier(self):
        return GaussianNB()

from typing import List

from sklearn.neighbors import KNeighborsClassifier

from learn.utils import LearnUtils


class KNN:
    n_neighbors = 7

    def learn(self) -> List[int]:
        labels, train_array, test_array = LearnUtils.get_learn_data()

        clf = KNeighborsClassifier(self.n_neighbors)
        clf.fit(train_array, labels)

        return clf.predict(test_array).tolist()

    def get_classifier(self):
        return KNeighborsClassifier(self.n_neighbors)

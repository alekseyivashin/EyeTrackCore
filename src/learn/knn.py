from typing import List

from sklearn import neighbors

from learn.utils import LearnUtils


class KNN:
    n_neighbors = 7

    def learn(self) -> List[int]:
        labels, train_array, test_array = LearnUtils.get_learn_data()

        clf = neighbors.KNeighborsClassifier(self.n_neighbors)
        clf.fit(train_array, labels)

        Z = clf.predict(test_array).tolist()
        return LearnUtils.prediction_to_result(Z)

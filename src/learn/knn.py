from typing import List

from sklearn import neighbors

from learn.utils import vector_list_to_array
from learn.vector import Vector


class KNN:
    n_neighbors = 2

    def learn(self, train_vectors: List[Vector], test_vectors: List[Vector], labels: List[str]):
        train_array = vector_list_to_array(train_vectors)
        test_array = vector_list_to_array(test_vectors)

        clf = neighbors.KNeighborsClassifier(self.n_neighbors)
        clf.fit(train_array, labels)

        Z = clf.predict(test_array)
        a = 1

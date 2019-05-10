from typing import List, Dict

from sklearn.cluster import KMeans

from learn.utils import LearnUtils
from learn.vector import Vector


class KMeansMethod:

    def learn(self) -> List[int]:
        labels, train_array, test_array = LearnUtils.get_learn_data()

        clf = KMeans(n_clusters=LearnUtils.get_labels_count())
        clf.fit(train_array, labels)

        Z = clf.predict(test_array).tolist()
        return LearnUtils.prediction_to_result(Z)
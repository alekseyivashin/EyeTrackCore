from typing import List

from sklearn.svm import SVC

from learn.utils import LearnUtils


class SVCMethod:

    def learn(self) -> List[int]:
        labels, train_array, test_array = LearnUtils.get_learn_data()

        clf = SVC(kernel="poly")
        clf.fit(train_array, labels)

        Z = clf.predict(test_array).tolist()
        return LearnUtils.prediction_to_result(Z)
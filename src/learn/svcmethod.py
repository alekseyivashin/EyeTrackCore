from typing import List

from sklearn.svm import SVC

from learn.utils import LearnUtils


class SVCMethod:

    def learn(self) -> List[int]:
        labels, train_array, test_array = LearnUtils.get_learn_data()

        clf = SVC(kernel="poly")
        clf.fit(train_array, labels)

        return clf.predict(test_array).tolist()

    def get_classifier(self):
        return SVC(kernel="poly")

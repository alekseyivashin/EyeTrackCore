from typing import List

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from learn.utils import LearnUtils


class Voting:
    n_neighbors = 7

    def learn(self) -> List[int]:
        labels, train_array, test_array = LearnUtils.get_learn_data()
        clf = self.__create_classifier()
        clf.fit(train_array, labels)
        return clf.predict(test_array).tolist()

    def cross_validation(self) -> List[float]:
        labels, train_array = LearnUtils.get_cross_val_data()
        clf = self.__create_classifier()
        return cross_val_score(clf, train_array, labels, cv=6)

    def __create_classifier(self):
        knn = KNeighborsClassifier(self.n_neighbors)
        svc = SVC(kernel="poly", probability=True)
        random_forest = RandomForestClassifier()
        gaussian = GaussianProcessClassifier()
        bayes = GaussianNB()

        return VotingClassifier(
            estimators=[("knn", knn), ("svc", svc), ("rf", random_forest), ("gaus", gaussian), ("bayes", bayes)],
            voting="soft")

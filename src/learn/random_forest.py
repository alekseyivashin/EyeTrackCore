from typing import List

from sklearn.ensemble import RandomForestClassifier

from learn.utils import LearnUtils


class RandomForest:

    def learn(self) -> List[int]:
        labels, train_array, test_array = LearnUtils.get_learn_data()

        clf = RandomForestClassifier()
        clf.fit(train_array, labels)

        Z = clf.predict(test_array).tolist()
        return LearnUtils.prediction_to_result(Z)
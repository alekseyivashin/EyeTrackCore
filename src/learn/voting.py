from typing import List

from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectFpr, SelectFdr, SelectFwe
from sklearn.model_selection import cross_val_score

from learn.bayes import Bayes
from learn.knn import KNN
from learn.random_forest import RandomForest
from learn.svcmethod import SVCMethod
from learn.utils import LearnUtils


class Voting:
    __n_neighbors = 7
    __feature_methods = {
        "k_chi_3": SelectKBest(chi2, k=3),
        "k_chi_4": SelectKBest(chi2, k=4),
        "k_chi_5": SelectKBest(chi2, k=5),
        "k_fclassif_3": SelectKBest(f_classif, k=3),
        "k_fclassif_4": SelectKBest(f_classif, k=4),
        "k_fclassif_5": SelectKBest(f_classif, k=5),
        "k_mutual_3": SelectKBest(mutual_info_classif, k=3),
        "k_mutual_4": SelectKBest(mutual_info_classif, k=4),
        "k_mutual_5": SelectKBest(mutual_info_classif, k=5),
        "fpr_chi_01": SelectFpr(chi2, alpha=0.1),
        "fpr_chi_005": SelectFpr(chi2, alpha=0.05),
        "fpr_chi_001": SelectFpr(chi2, alpha=0.01),
        "fpr_fclassif_01": SelectFpr(f_classif, alpha=0.1),
        "fpr_fclassif_005": SelectFpr(f_classif, alpha=0.05),
        "fpr_fclassif_001": SelectFpr(f_classif, alpha=0.01),
        "fnr_chi_01": SelectFdr(chi2, alpha=0.1),
        "fnr_chi_005": SelectFdr(chi2, alpha=0.05),
        "fnr_chi_001": SelectFdr(chi2, alpha=0.01),
        "fnr_fclassif_01": SelectFdr(f_classif, alpha=0.1),
        "fnr_fclassif_005": SelectFdr(f_classif, alpha=0.05),
        "fnr_fclassif_001": SelectFdr(f_classif, alpha=0.01),
        "fwe_chi_01": SelectFwe(chi2, alpha=0.1),
        "fwe_chi_005": SelectFwe(chi2, alpha=0.05),
        "fwe_chi_001": SelectFwe(chi2, alpha=0.01),
        "fwe_fclassif_01": SelectFwe(f_classif, alpha=0.1),
        "fwe_fclassif_005": SelectFwe(f_classif, alpha=0.05),
        "fwe_fclassif_001": SelectFwe(f_classif, alpha=0.01),
    }

    def get_feature_method_names(self):
        return self.__feature_methods.keys()

    def learn(self, feature_method_name: str = None) -> List[int]:
        labels, train_array, test_array = LearnUtils.get_learn_data()
        if feature_method_name is not None:
            feature_filter = self.__feature_methods[feature_method_name]
            feature_filter.fit(train_array, labels)
            train_array = feature_filter.transform(train_array)
            test_array = feature_filter.transform(test_array)
        clf = self.__create_classifier()
        clf.fit(train_array, labels)
        return clf.predict(test_array).tolist()

    def cross_validation(self) -> List[float]:
        labels, train_array = LearnUtils.get_cross_val_data()
        clf = self.__create_classifier()
        return cross_val_score(clf, train_array, labels, cv=6)

    def __create_classifier(self):
        knn = KNN().get_classifier()
        svc = SVCMethod().get_classifier()
        random_forest = RandomForest().get_classifier()
        bayes = Bayes().get_classifier()

        return VotingClassifier(
            estimators=[
                ("knn", knn),
                ("svc", svc),
                ("rf", random_forest),
                ("bayes", bayes)
            ],
            voting="soft")

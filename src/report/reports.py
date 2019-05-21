from typing import List

from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

from learn.utils import LearnUtils


def print_classification_report(encoded_labels, result, labels):
    print(classification_report(encoded_labels, result, target_names=labels))


def get_average_precision_score(y_test, y_score):
    classes = LearnUtils.get_encoded_labels()
    y_test = label_binarize(y_test, classes=classes)
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,
                                                         average="micro")
    return precision, recall, average_precision

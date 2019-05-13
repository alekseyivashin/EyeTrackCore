from typing import List

from sklearn.metrics import classification_report


def print_classification_report(encoded_labels, result, labels):
    print(classification_report(encoded_labels, result, target_names=labels))
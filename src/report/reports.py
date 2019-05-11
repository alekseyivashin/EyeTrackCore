from typing import List

from sklearn.metrics import classification_report


def print_classification_report(encoded_labels: List[int], major_result: List[int], labels: List[str]):
    print(classification_report(encoded_labels, major_result, target_names=labels))
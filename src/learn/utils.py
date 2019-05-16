from typing import List, Dict, Tuple, Any

from numpy import ndarray, array
from sklearn.preprocessing import scale, LabelEncoder

from learn.vector import Vector


class LearnUtils:
    __vectors: Dict[str, List[Vector]]
    __labels_encoder: LabelEncoder
    __labels: List[str]
    __labels_count: int
    __encoded_labels: Any
    __encoded_labels_for_train: Any
    __train_array: ndarray
    __test_array: ndarray

    @staticmethod
    def set_up(vectors: Dict[str, List[Vector]], test_indexes: List[int]):
        LearnUtils.__vectors = vectors
        LearnUtils.__labels_encoder = LabelEncoder()
        LearnUtils.__labels = list(LearnUtils.__vectors.keys())
        LearnUtils.__labels_encoder.fit(LearnUtils.__labels)
        LearnUtils.__labels_count = len(LearnUtils.__labels)
        LearnUtils.__encoded_labels = LearnUtils.__labels_encoder.transform(list(LearnUtils.__vectors.keys()))

        labels_for_train, train_array, test_array = LearnUtils.__preprocess_data(LearnUtils.__vectors, test_indexes)
        LearnUtils.__encoded_labels_for_train = LearnUtils.__labels_encoder.transform(labels_for_train)
        LearnUtils.__train_array = scale(train_array)
        LearnUtils.__test_array = scale(test_array)

    @staticmethod
    def get_learn_data() -> Tuple[Any, ndarray, ndarray]:
        return LearnUtils.__encoded_labels_for_train, LearnUtils.__train_array, LearnUtils.__test_array

    @staticmethod
    def get_cross_val_data() -> Tuple[Any, ndarray]:
        labels, train_array, test_array = LearnUtils.__preprocess_data(LearnUtils.__vectors)
        return labels, scale(train_array)

    @staticmethod
    def get_labels() -> List[str]:
        return LearnUtils.__labels

    @staticmethod
    def get_encoded_labels() -> List[int]:
        return LearnUtils.__encoded_labels.tolist()

    @staticmethod
    def decode_labels(encoded_labels: List[int]) -> List[str]:
        return LearnUtils.__labels_encoder.inverse_transform(encoded_labels)

    @staticmethod
    def get_labels_count() -> int:
        return LearnUtils.__labels_count

    @staticmethod
    def __vector_list_to_array(vectors: List[Vector]) -> ndarray:
        return array([vector.to_array() for vector in vectors])

    @staticmethod
    def __preprocess_data(vectors: Dict[str, List[Vector]], test_indexes: List[int] = None) -> Tuple[Any, ndarray, ndarray]:
        labels = []
        train_vectors = []
        test_vectors = []
        for name in vectors:
            for i, vector in enumerate(vectors[name]):
                if test_indexes is not None and i in test_indexes:
                    test_vectors.append(vector)
                else:
                    train_vectors.append(vector)
                    labels.append(name)
        return labels, LearnUtils.__vector_list_to_array(train_vectors), LearnUtils.__vector_list_to_array(test_vectors)

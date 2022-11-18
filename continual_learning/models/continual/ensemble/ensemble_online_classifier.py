from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool
from pprint import pprint
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from river.ensemble import AdaptiveRandomForestClassifier, AdaBoostClassifier
from river.tree import HoeffdingTreeClassifier, ExtremelyFastDecisionTreeClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
from tqdm import tqdm

from continual_learning.datasets.mnist import (
    load_mnist_from_pickle,
    DOMAIN_INCREMENTAL_MNIST_TRAIN_PICKLE_PATH,
    DOMAIN_INCREMENTAL_MNIST_TEST_PICKLE_PATH
)


def learn_one(model, x, y):
    model.learn_one(x, y)
    return model


class EnsembleOnlineClassifier:
    """Adaptation of https://arxiv.org/pdf/2105.13327.pdf for online learning models (e.g.
    HoeffdingTrees).
    :param keys: memory size x key size
    """
    def __init__(self, keys: np.ndarray, k_neighbors: int, size: int, soft_voting: bool = True):
        self.keys = keys
        self._normalized_keys = normalize(keys)
        self.k_neighbors = k_neighbors
        self.size = size
        self.soft_voting = soft_voting
        self.models = self._init_models()
        self._seen_classes = set()
        self._models_stats = defaultdict(lambda: defaultdict(int))

    def lookup_memory(self, input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cos_similarity = cosine_similarity(self.keys, input)
        indexes = np.argsort(cos_similarity, axis=0)[::-1][:self.k_neighbors].flatten()
        cos_similarity = cos_similarity[indexes]
        return indexes, cos_similarity

    def _init_models(self) -> np.ndarray:
        return np.array([HoeffdingTreeClassifier() for _ in range(self.size)])

    def _is_new_class_in_batch(self, labels: np.ndarray) -> bool:
        return not set(np.unique(labels)).issubset(self._seen_classes)

    def train(self, examples: np.ndarray, labels: np.ndarray) -> None:
        for example, label in tqdm(zip(examples, labels), total=len(labels), desc=f"Training ensemble online model"):
            self._train_single(example, label)

    def _train_single(self, example, label) -> None:
        example = np.atleast_2d(example)
        x = _array_to_dict(example.flatten())

        if label in self._seen_classes:
            indexes, _ = self.lookup_memory(example)
        else:
            self._seen_classes.add(label)
            indexes = list(range(len(self.models)))

        for index in indexes:
            self._models_stats[index][label] += 1

        map_results = []
        pool = Pool()
        for index in indexes:
            map_results.append(pool.apply_async(learn_one, args=(self.models[index], x, label)))
            # self.models[index].learn_one(x, label)
        pool.close()
        pool.join()
        for map_result_index, index in enumerate(indexes):
            self.models[index] = map_results[map_result_index].get()

    def predict(self, examples: np.ndarray) -> np.ndarray:
        predictions = []
        for example in tqdm(examples, total=len(examples), desc=f"Predicting..."):
            example = np.atleast_2d(example)
            indexes, cos_similarity = self.lookup_memory(example)
            example_predictions = []
            x = _array_to_dict(example.flatten())
            for model in self.models[indexes]:
                model_prediction = model.predict_one(x)
                example_predictions.append(model_prediction)
            if self.soft_voting:
                soft_votes = [0] * len(self._seen_classes)
                for cls, sim in zip(example_predictions, cos_similarity):
                    soft_votes[cls] += sim[0]
                predictions.append(np.argmax(soft_votes))
            else:
                counter = Counter(example_predictions)
                predictions.append(max(counter, key=counter.get))

        return np.array(predictions)

    def draw_key_space(self):
        total_samples = []
        dominating_classes = []
        for index in range(len(self.models)):
            model_metadata = self._models_stats[index]
            total_samples.append(sum(model_metadata.values()))
            dominating_classes.append(max(model_metadata, key=model_metadata.get))

        tsne = TSNE()
        embeddings = tsne.fit_transform(self.keys)
        x, y = embeddings[:, 0], embeddings[:, 1]
        ax = sns.scatterplot(x=x, y=y, hue=dominating_classes, size=total_samples, palette='RdYlGn')
        for index, (x_coord, y_coord, samples) in enumerate(zip(x, y, total_samples)):
            ax.annotate(samples, (x_coord, y_coord))

        plt.show()


def _array_to_dict(arr: np.ndarray) -> Dict[str, float]:
    return dict(zip(range(len(arr)), arr))


if __name__ == '__main__':
    n = 5

    classes = [0, 1, 1, 2, 2]
    cos_similarity = cosine_similarity([[-1, -0.5, -2], [1, 0, -1], [1, 0, -2], [-1, 0, -2], [-1, 0, -3]], [[-1,-1, 0]])
    indexes = np.argsort(cos_similarity, axis=0)[::-1][:n].flatten()
    cos_similarity = cos_similarity[indexes].flatten()
    print(indexes)
    print(cos_similarity[indexes])

    counter = Counter(classes)
    print(f"Hard voted: {max(counter, key=counter.get)}")

    soft_votes = [0] * len(cos_similarity)
    for cls, sim in zip(classes, cos_similarity):
        soft_votes[cls] += sim

    print(f"Soft voted: {soft_votes}, {np.argmax(soft_votes)}")

    # Classifier
    count = 200
    raw_X_train, raw_y_train, raw_X_test, raw_y_test = load_mnist_from_pickle(
        DOMAIN_INCREMENTAL_MNIST_TRAIN_PICKLE_PATH,
        DOMAIN_INCREMENTAL_MNIST_TEST_PICKLE_PATH
    )

    # keys = np.random.uniform(low=-1, high=1, size=(100, 784))
    keys = np.random.normal(size=(100, 784))
    classifier = EnsembleOnlineClassifier(keys, k_neighbors=10, size=len(keys), soft_voting=True)
    classifier.train(raw_X_train[:count].values, raw_y_train[:count].values)
    classifier.draw_key_space()
    y_pred = classifier.predict(raw_X_test[:count].values)
    y_test = raw_y_test[:count].values

    print(set(raw_y_train[:count].values), set(y_test))

    pprint(classification_report(y_test, y_pred, output_dict=True)['macro avg'])

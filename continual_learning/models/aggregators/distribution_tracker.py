from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from river.proba import Gaussian
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE


class ClassDistributionTracker:
    def __init__(self, attributes_count: int):
        self.attributes_count = attributes_count
        self.data = {}

    def add(self, items: np.ndarray, labels: np.ndarray) -> None:
        assert len(items) == len(labels)

        for example, label in zip(items, labels):
            if label not in self.data:
                self.data[label] = [Gaussian() for _ in range(self.attributes_count)]
            for index, value in enumerate(example):
                self.data[label][index].update(value)

    def sample(self, labels: np.ndarray):
        values = []
        for label in labels:
            mu, sigma = self.get_mu_sigma_values(label)
            sample = np.random.normal(mu, sigma)
            values.append(sample)

        return np.array(values)

    def get_mu_sigma_values(self, label) -> Tuple[np.ndarray, np.ndarray]:
        mu, sigma = [], []
        for attribute_dist in self.data[label]:
            mu.append(attribute_dist.mu)
            sigma.append(attribute_dist.sigma)

        return np.array(mu), np.array(sigma)


if __name__ == '__main__':
    n_features = 50
    n_classes = 2
    n_samples = 500

    x, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes)
    tracker = ClassDistributionTracker(attributes_count=n_features)

    tracker.add(x, y)

    labels_to_sample = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    samples = tracker.sample(labels_to_sample)

    x = np.concatenate([x, samples])
    y = np.concatenate([y, labels_to_sample])

    import seaborn as sns
    tnse = TSNE()
    x_tnse = tnse.fit_transform(x)
    hue = (['Data'] * n_samples) + (['Generated'] * len(labels_to_sample))
    sns.scatterplot(x=x_tnse[:, 0], y=x_tnse[:, 1], hue=hue, style=y)
    plt.savefig('Samples.png')

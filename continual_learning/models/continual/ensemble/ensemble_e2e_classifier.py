from collections import Counter
from pprint import pprint
from typing import Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from avalanche.benchmarks import SplitMNIST
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from torch import Tensor
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from tqdm import tqdm

from continual_learning.models.autoencoder.omniglot import OmniglotAutoencoderModel, \
    OmniglotAutoencoder
from continual_learning.models.continual.ensemble.weak_learner.group import WeakLearnersGroup
from continual_learning.models.continual.ensemble.weak_learner.neural_network_pytorch import \
    NeuralNetworkPyTorch
from continual_learning.models.continual.ensemble.weak_learner.neural_network_sklearn import \
    NeuralNetworkSklearn
from settings import MODELS_DIR


class EnsembleE2EClassifier:
    """Adaptation of https://arxiv.org/pdf/2105.13327.pdf for end-to-end continual learning.
    :param keys: memory size x key size
    """
    def __init__(self,
        keys: np.ndarray,
        k_neighbors: int,
        weak_learners_group: WeakLearnersGroup,
        encoder: OmniglotAutoencoder,
        classes: List[int],
        soft_voting: bool = True,
        tqdm_enabled: bool = False,
    ):
        self.keys = normalize(keys, norm='l2', copy=False)
        self.k_neighbors = k_neighbors
        self.models_group = weak_learners_group
        self.encoder = encoder
        self.classes = classes
        self.soft_voting = soft_voting
        self.tqdm_enabled = tqdm_enabled
        self._normalized_keys = normalize(keys)

        self.encoder.freeze()

    def lookup_memory(self, input: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        input = normalize(input, norm='l2', copy=False)
        cos_similarity = self.keys @ input.T
        indexes = np.argsort(cos_similarity, axis=0)[::-1][:self.k_neighbors].flatten()  # TODO take last k_neighbors values
        cos_similarity = cos_similarity[indexes]
        mean_key = np.mean(self.keys[indexes], axis=0)
        return indexes, cos_similarity, mean_key

    def fit(self, examples: Tensor, labels: Tensor) -> None:
        for example, label in tqdm(zip(examples, labels), total=len(labels), disable=not self.tqdm_enabled, desc=f"Training ensemble online model"):
            self._train_single(example, label)

    def predict(self, examples: Tensor, soft_voting: Optional[bool] = None) -> np.ndarray:
        y_pred = []
        for example in tqdm(examples, total=len(examples), disable=not self.tqdm_enabled, desc=f"Predicting..."):
            y_pred.append(self._predict_single(example, soft_voting=soft_voting))

        return np.array(y_pred)

    def encode(self, example: Tensor) -> Tensor:
        example = example.reshape((1, 1, 28, 28))
        joint_encoding, _, _ = self.encoder.model.encode(example)
        return joint_encoding

    def _train_single(self, example: Tensor, label: Tensor) -> None:
        assert example.dim() == 2

        if not self.models_group.is_known_class(label):
            self.models_group.update_seen_classes(label)

        example = self.encode(example)

        indexes, cos_similarity, mean_key = self.lookup_memory(example.numpy())
        clf_meta = {'cos_similarity': cos_similarity, 'mean_key': mean_key}

        self.models_group.fit(example, label, indexes, clf_meta=clf_meta)
        self.models_group.update_model_stats(indexes, label.tolist())

    def _predict_single(self, example: Tensor, soft_voting: Optional[bool] = None) -> np.ndarray:
        assert example.dim() == 2

        example = self.encode(example)

        indexes, cos_similarity, mean_key = self.lookup_memory(example.numpy())
        clf_meta = {'cos_similarity': cos_similarity, 'mean_key': mean_key}
        example_predictions = self.models_group.predict(example, indexes, clf_meta=clf_meta)

        can_vote_soft = self.soft_voting if soft_voting is None else soft_voting
        if can_vote_soft:
            soft_votes = [0] * len(self.classes)
            for cls, sim in zip(example_predictions, cos_similarity):
                soft_votes[cls] += sim[0]
            return np.argmax(soft_votes)
        else:
            counter = Counter(example_predictions)
            return max(counter, key=counter.get).item()

    def draw_key_space(self):
        try:
            total_samples = []
            dominating_classes = []
            for index in range(self.models_group.size):
                model_metadata = self.models_group.models_stats[index]
                total_samples.append(sum(model_metadata.values()))
                dominating_classes.append(max(model_metadata, key=model_metadata.get))

            tsne = TSNE()
            embeddings = tsne.fit_transform(self.keys)
            x, y = embeddings[:, 0], embeddings[:, 1]
            ax = sns.scatterplot(x=x, y=y, hue=dominating_classes, size=total_samples, palette='RdYlGn')
            for index, (x_coord, y_coord, samples) in enumerate(zip(x, y, total_samples)):
                ax.annotate(samples, (x_coord, y_coord))

            plt.savefig("Keyspace.png")
        except Exception as exception:
            print(f"Cannot draw key space ({exception})")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    ENCODER_OUTPUT_PATH = MODELS_DIR / 'ensemble_omniglot_autoencoder' / 'encoder.ckpt'

    num_classifiers = 128
    n_neighbors = 10
    sample_size = 512
    ensemble_learning_rate = 0.0001

    # Sanity check
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    group = WeakLearnersGroup(
        size=num_classifiers,
        weak_learner=NeuralNetworkPyTorch,
        # Weak learner args
        classes=classes,
        learning_rate=ensemble_learning_rate,
        input_size=sample_size,
    )

    # Keys
    # keys = np.random.uniform(low=-1, high=1, size=(num_classifiers, sample_size))
    keys = np.random.normal(size=(num_classifiers, sample_size))

    # Autoencoder
    autoencoder = OmniglotAutoencoder.load_from_checkpoint(
        checkpoint_path=ENCODER_OUTPUT_PATH,
        input_size=28,
        encoder_size=sample_size,
        learning_rate=0.001,
    )

    # Classifier
    classifier = EnsembleE2EClassifier(
        keys,
        k_neighbors=n_neighbors,
        weak_learners_group=group,
        encoder=autoencoder,
        classes=classes,
        soft_voting=True,
    )

    transform_mnist = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
    ])
    scenario = SplitMNIST(
        n_experiences=10,
        return_task_id=False,
        fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        train_transform=transform_mnist,
        eval_transform=transform_mnist,
    )

    train_stream = scenario.train_stream
    test_stream = scenario.test_stream

    # Train
    limit = 1000
    for exp_num, experience in enumerate(train_stream):
        current = 0
        dataset = experience.dataset
        for x, y, task in tqdm(dataset, total=len(dataset), desc=f"Training experience {exp_num}"):
            y = torch.LongTensor([y])
            classifier.fit(x, y)

            current += 1
            if limit is not None and current > limit:
                break

    # Test
    limit = 100

    y_pred, y_test = [], []
    for exp_num, experience in enumerate(test_stream):
        current = 0

        dataset = experience.dataset
        for x, y, task in tqdm(dataset, total=len(dataset), desc=f"Evaluating experience {exp_num}"):
            y_pred_single = classifier.predict(x)
            y_pred.extend(y_pred_single)
            y_test.append(y)

            current += 1
            if limit is not None and current > limit:
                break

    # classifier.draw_key_space()
    pprint(classifier.models_group.models_stats)
    pprint(classification_report(y_test, y_pred, output_dict=False))

    plt.clf()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True)
    plt.savefig("Confusion_matrix.png")

from pathlib import Path
from pprint import pprint
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from avalanche.benchmarks import SplitMNIST
from sklearn.metrics import classification_report, confusion_matrix
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from continual_learning.models.autoencoder.omniglot import OmniglotAutoencoder
from continual_learning.models.continual.ensemble.ensemble_e2e_module import EnsembleE2EModule, \
    generate_batch
from settings import MODELS_DIR


if __name__ == '__main__':
    import warnings
    # TODO fix warnings
    warnings.filterwarnings("ignore")  # ".*does not have many workers.*"

    ENCODER_OUTPUT_PATH = MODELS_DIR / 'ensemble_omniglot_autoencoder' / 'encoder.ckpt'

    num_classifiers = 128
    n_neighbors = 8
    sample_size = 512
    ensemble_learning_rate = 0.0001
    ensemble_weight_decay = 0.0001
    tanh_factor = 250.0
    trainable_keys = False
    augment_data = False

    num_generated_batches = 1
    num_generated_examples = 20

    batch_size = 60
    train_limit = None
    test_limit = None

    # Sanity check
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    import random
    random.shuffle(classes)

    # Keys
    # keys = np.random.uniform(low=-1, high=1, size=(num_classifiers, sample_size))
    keys = np.random.normal(size=(num_classifiers, sample_size))
    keys = torch.from_numpy(keys)
    keys = keys.type(torch.FloatTensor)

    # Autoencoder
    autoencoder = OmniglotAutoencoder.load_from_checkpoint(
        checkpoint_path=ENCODER_OUTPUT_PATH,
        input_size=28,
        encoder_size=sample_size,
        learning_rate=0.001,
    )

    # Classifier
    classifier = EnsembleE2EModule(
        encoder=autoencoder,
        input_size=sample_size,
        num_classifiers=num_classifiers,
        keys=keys,
        k_neighbors=n_neighbors,
        classes=classes,
        learning_rate=ensemble_learning_rate,
        weight_decay=ensemble_weight_decay,
        tanh_factor=tanh_factor,
        hard_voting=False,
        freeze_encoder=True,
        trainable_keys=trainable_keys,
    )

    transform_mnist = Compose([
        ToTensor(),
    ])
    scenario = SplitMNIST(
        n_experiences=len(classes),
        return_task_id=False,
        fixed_class_order=classes,
        train_transform=transform_mnist,
        eval_transform=transform_mnist,
    )

    train_stream = scenario.train_stream
    test_stream = scenario.test_stream

    # Train
    for exp_num, experience in enumerate(train_stream):
        current = 0
        dataset = experience.dataset
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for x, y, task in tqdm(data_loader, total=len(data_loader), desc=f"Training experience {exp_num}"):
            y = torch.LongTensor(y)
            classifier.fit(x, y)

            current += 1
            if train_limit is not None and current > train_limit:
                break

        if augment_data:
            for _ in range(num_generated_batches):
                # Training on additional generated batch
                x, y = generate_batch(classifier.distribution_tracker, {seen_class: num_generated_examples for seen_class in classifier.seen_classes})
                classifier.fit(x, y, x_is_encoded=True)

    # Test
    y_pred_vanilla, y_pred_tanh, y_pred_ensemble, y_test = [], [], [], []
    misclasified = []
    for exp_num, experience in enumerate(test_stream):
        current = 0
        dataset = experience.dataset
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for x, y, task in tqdm(data_loader, total=len(data_loader), desc=f"Evaluating experience {exp_num}"):
            y_preds_dict = classifier.predict(x, return_dict=True)
            y_pred_vanilla.extend(y_preds_dict['vanilla'].numpy())
            y_pred_tanh.extend(y_preds_dict['tanh'].numpy())
            y_pred_ensemble.extend(y_preds_dict['ensemble'].numpy())
            y_test.extend(y.numpy())

            current += 1
            if test_limit is not None and current > test_limit:
                break

    classifier.plot_loss()

    # classifier.draw_key_space()
    pprint(classifier.models_stats)
    print("Vanilla classifier")
    pprint(classification_report(y_test, y_pred_vanilla, output_dict=True))
    print("Tanh classifier")
    pprint(classification_report(y_test, y_pred_tanh, output_dict=True))
    print("Ensemble classifier")
    pprint(classification_report(y_test, y_pred_ensemble, output_dict=True))

    plt.clf()
    cm = confusion_matrix(y_test, y_pred_vanilla)
    sns.heatmap(cm, annot=True, fmt='')
    plt.savefig("Confusion_matrix_vanilla.png")

    plt.clf()
    cm = confusion_matrix(y_test, y_pred_tanh)
    sns.heatmap(cm, annot=True, fmt='')
    plt.savefig("Confusion_matrix_tanh.png")

    plt.clf()
    cm = confusion_matrix(y_test, y_pred_ensemble)
    sns.heatmap(cm, annot=True, fmt='')
    plt.savefig("Confusion_matrix_ensemble.png")

    print('Updates', classifier.optimizer.updates_count)
    print('Grad none',  classifier.optimizer.grad_none)

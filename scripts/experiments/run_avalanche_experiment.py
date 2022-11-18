import json
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Type, List

import torch
from avalanche.benchmarks import PermutedMNIST, SplitCIFAR100
from avalanche.evaluation.metrics import loss_metrics, accuracy_metrics, timing_metrics, \
    cpu_usage_metrics, ExperienceForgetting, StreamConfusionMatrix, disk_usage_metrics, \
    ExperienceBWT, forgetting_metrics, bwt_metrics
from avalanche.logging import InteractiveLogger, TextLogger, CSVLogger
from avalanche.training import Naive, BaseStrategy, GDumb, EWC, GEM, LwF, AGEM, \
    SynapticIntelligence, LFL, CWRStar, Replay
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10
from avalanche.models import SimpleMLP, MTSimpleMLP, SimpleCNN

# Config

from settings import EXPERIMENTS_OUTPUT_DIR

REPEAT = 5
LR = 0.0005
N_CLASSES = 10
N_EXPERIENCES = 5
HIDDEN_SIZE = 128
INPUT_SIZE = 28 * 28

MNIST_CIFAR10_EXPERIENCES = 10
CIFAR_100_EXPERIENCES = 100


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

shared_params = dict(train_mb_size=32, train_epochs=1, eval_mb_size=32)

# --- HP ---
# SimpleMLP(num_classes=10, hidden_size=750, drop_rate=0.0) EWC ewc_lambda=100. lr=0.000001

strategies = {
    # 'Naive': (Naive, dict(**shared_params)),
    'LwF': (LwF, dict(alpha=0.15, temperature=1.5, **shared_params)),
    # 'GDumb': (GDumb, dict(mem_size=200, **shared_params)),

    # 'EWC_1': (EWC, dict(ewc_lambda=.1, **shared_params)),
    # 'EWC_2': (EWC, dict(ewc_lambda=1., **shared_params)),
    # 'EWC_3': (EWC, dict(ewc_lambda=10., **shared_params)),
    # 'EWC_4': (EWC, dict(ewc_lambda=100., **shared_params)),
    # 'EWC_5': (EWC, dict(ewc_lambda=1000., **shared_params)),
    'EWC': (EWC, dict(ewc_lambda=10000., **shared_params)),
    'GEM': (GEM, dict(patterns_per_exp=100, memory_strength=0.5, **shared_params)),

    # 'CWRStar': (CWRStar, dict(cwr_layer_name=None, **shared_params)),
    # 'Replay': (Replay, dict(mem_size=200, **shared_params)),
    # 'AGEM': (AGEM, dict(patterns_per_exp=200, sample_size=64, **shared_params)),
    # 'SynapticIntelligence': (SynapticIntelligence, dict(si_lambda=0.0005)),
    # 'LFL': (LFL, dict(lambda_e=0.1, **shared_params))
    # 'GSS_greedy': (),
    # 'CoPE': (),
}


def create_default_dirname(prefix: str = 'experiment') -> str:
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    return f"{prefix}-{current_time}"


def create_default_datetime_filename(prefix: str = 'experiment_log', extension: str = 'txt') -> str:
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    return f"{prefix}-{current_time}.{extension}"


def get_plugin(log_folder: Path, n_classes: int = 10) -> Any:
    logging_filepath = log_folder / create_default_datetime_filename()
    text_logger = TextLogger(open(logging_filepath, 'a'))
    interactive_logger = InteractiveLogger()
    csv_logger = CSVLogger(log_folder)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=False, experience=False, stream=True),
        loss_metrics(minibatch=False, epoch=False, experience=False, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        cpu_usage_metrics(experience=False),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        # ExperienceForgetting(),
        # ExperienceBWT(),
        # StreamConfusionMatrix(num_classes=n_classes, save_image=False),
        # disk_usage_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loggers=[
            # interactive_logger
        ],
    )

    return eval_plugin


def setup_training_sessions(strategy: Type[BaseStrategy], strategy_kwargs: Dict[str, Any],
                            device: torch.device, log_folder: Path, dataset_name: str):
    return [setup_training(strategy, strategy_kwargs, device, log_folder, dataset_name) for _ in range(REPEAT)]


def setup_training(strategy: Type[BaseStrategy], strategy_kwargs: Dict[str, Any],
                   device: torch.device, log_folder: Path, dataset_name: str):
    # model
    if dataset_name == 'mnist':
        model = SimpleMLP(num_classes=10, input_size=INPUT_SIZE, hidden_size=512)
        dataset = SplitMNIST(n_experiences=MNIST_CIFAR10_EXPERIENCES, return_task_id=False)
    elif dataset_name == 'cifar10':
        dataset = SplitCIFAR10(n_experiences=MNIST_CIFAR10_EXPERIENCES, return_task_id=False)
        model = SimpleCNN(num_classes=10)
    else:
        dataset = SplitCIFAR100(n_experiences=CIFAR_100_EXPERIENCES, return_task_id=False)
        model = SimpleCNN(num_classes=100)

    train_stream = dataset.train_stream
    test_stream = dataset.test_stream

    # Prepare for training & testing
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()

    # Prepare for evaluation
    evaluator = get_plugin(n_classes=N_CLASSES, log_folder=log_folder)

    # Continual learning strategy
    cl_strategy = strategy(model, optimizer, criterion, device=device, evaluator=evaluator,
                           **strategy_kwargs)

    metadata = {'learning_rate': LR}

    return train_stream, test_stream, cl_strategy, metadata


if __name__ == '__main__':
    log_folder = EXPERIMENTS_OUTPUT_DIR / create_default_dirname()
    log_folder.mkdir(parents=True, exist_ok=True)

    print(f"Logging into folder: {log_folder}")

    results = defaultdict(lambda: defaultdict(list))

    for dataset_name in ['mnist', 'cifar10', 'cifar100']:
        print(f"Running {dataset_name} experiments")
        for strategy_name, (strategy, strategy_kwargs) in tqdm(strategies.items(), total=len(strategies)):
            for index, (train_stream, test_stream, cl_strategy, metadata) in enumerate(setup_training_sessions(strategy, strategy_kwargs, device, log_folder, dataset_name)):
                results_key = f"{dataset_name}_{strategy_name}_{index}"
                print(f"Running {results_key}")
                try:
                    for train_task in train_stream:
                        cl_strategy.train(train_task, num_workers=6)
                        results[results_key]['data'].append(cl_strategy.eval(test_stream))
                        results[results_key]['metadata'] = {**metadata, **strategy_kwargs}

                        with open(log_folder / 'experiment_log.json', 'w') as fp:
                            json.dump(results, fp)
                except Exception as exp:
                    print(f"Cannot train {results_key}", exp)

    print('======================== Results ========================')
    print(results)

    with open(log_folder / 'experiment_log.json', 'w') as fp:
        json.dump(results, fp)

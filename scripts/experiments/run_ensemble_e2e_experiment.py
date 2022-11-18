import json
from collections import defaultdict
from pathlib import Path

from avalanche.evaluation.metrics import loss_metrics, timing_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from continual_learning.datasets.ensemble_e2e.dataset_loader import load_scenario
from continual_learning.experiments.ensemble_e2e.encoder_loader import load_encoder
from continual_learning.metrics.avalanche.accuracy import accuracy_metrics_from_strategy_state
from continual_learning.metrics.avalanche.backward_transfer import bwt_metrics_from_strategy_state
from continual_learning.metrics.avalanche.forgetting import forgetting_metrics_from_strategy_state
from continual_learning.models.continual.ensemble.ensemble_e2e_module_avalanche import \
    EnsembleE2EModule
from continual_learning.models.continual.ensemble.ensemble_e2e_pytorch_avalanche import \
    ensemble_e2e_loss, EnsembleE2E, create_default_dirname
from continual_learning.optimizers.naive_optimizer import NaiveOptimizer
from settings import EXPERIMENTS_OUTPUT_DIR


def setup_and_run(config: DictConfig, logging_dir: Path) -> None:
    import warnings
    warnings.filterwarnings("ignore")  # ".*does not have many workers.*"

    dataset = config.dataset

    # Scenario (dataset)
    scenario = load_scenario(dataset_name=dataset, config=config)

    # Encoder
    encoder = load_encoder(encoder_name=config.encoder, config=config)

    # Classifier
    classifier = EnsembleE2EModule(
        encoder=encoder,
        input_size=config.sample_size,
        num_classifiers=config.num_classifiers,
        k_neighbors=config.n_neighbors,
        learning_rate=config.ensemble_learning_rate,
        weight_decay=config.ensemble_weight_decay,
        tanh_factor=config.tanh_factor,
        hard_voting=False,
        trainable_keys=config.trainable_keys,
        num_classes=config.num_classes,
        classes=None,
        device=config.device,
    )

    # Optimizer
    optimizer = NaiveOptimizer(
        [
            {'params': classifier.vanilla_classifier.parameters()},
            {'params': classifier.tanh_classifier.parameters()},
            {'params': classifier.models.parameters()},
        ],
        learning_rate=config.ensemble_learning_rate,
        weight_decay=config.ensemble_weight_decay,
        device=config.device,
    )

    # Evaluation
    state_keys = ['vanilla', 'tanh', 'ensemble']
    experience_eval = False
    eval_plugin = EvaluationPlugin(
        # accuracy_metrics(minibatch=False, epoch=False, experience=experience_eval, stream=True),
        accuracy_metrics_from_strategy_state(state_keys=state_keys, minibatch=False, epoch=False,
                                             experience=experience_eval, stream=True),
        loss_metrics(minibatch=False, epoch=False, experience=experience_eval, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=experience_eval, stream=True),
        forgetting_metrics_from_strategy_state(state_keys=state_keys, experience=experience_eval,
                                               stream=True),
        bwt_metrics_from_strategy_state(state_keys=state_keys, experience=experience_eval,
                                        stream=True),
        # forward_transfer_metrics_from_strategy_state(state_keys=state_keys, experience=experience_eval, stream=True),
        # confusion_matrix_metrics(num_classes=num_classes, save_image=False, stream=True),
        loggers=[
            # InteractiveLogger()
        ],
        benchmark=scenario,
        strict_checks=False
    )

    # Strategy
    strategy = EnsembleE2E(
        model=classifier,
        optimizer=optimizer,
        criterion=ensemble_e2e_loss,
        evaluator=eval_plugin,
        # Kwargs
        train_mb_size=config.batch_size,
        train_epochs=config.train_epochs,
        eval_mb_size=config.batch_size,
        device=config.device,
        eval_every=-1
    )

    results = defaultdict(list)

    logging_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging into folder: {logging_dir}")

    train_stream = scenario.train_stream
    test_stream = scenario.test_stream
    for train_task in train_stream:
        strategy.train(train_task, num_workers=6)

        eval_result = strategy.eval(test_stream)
        results['data'].append(eval_result)
        results['knn_stats'].append({**classifier.dknn.run_meta})

        with open(logging_dir / 'config.yaml', 'w') as file:
            OmegaConf.save(config=conf, f=file.name)

        with open(logging_dir / 'experiment_log.json', 'w') as fp:
            json.dump(results, fp)


if __name__ == '__main__':
    common_config = {
        'ensemble_learning_rate': 0.0001, 'ensemble_weight_decay': 0.0001, 'tanh_factor': 250.0, 'trainable_keys': True,
        'device': 'cuda', 'train_epochs': 1, 'return_task_id': False, 'shuffle': True, 'augment_data': False,
        'num_generated_batches': 1, 'num_generated_examples': 20, 'fixed_class_order': None, 'setup': 'E2E'
    }

    configs_to_run = [
        # CIFAR 10
        {'dataset': 'cifar10', 'repeat': 5, **common_config, 'num_classifiers': 16, 'n_neighbors': 4, 'batch_size': 10,
         'num_experiences': 5, 'num_classes': 10, 'sample_size': 2048, 'encoder': 'byol'},
        {'dataset': 'cifar10', 'repeat': 5, **common_config, 'num_classifiers': 128, 'n_neighbors': 16,
         'batch_size': 10, 'num_experiences': 5, 'num_classes': 10, 'sample_size': 2048, 'encoder': 'byol'},
        {'dataset': 'cifar10', 'repeat': 5, **common_config, 'num_classifiers': 16, 'n_neighbors': 4, 'batch_size': 10,
         'num_experiences': 10, 'num_classes': 10, 'sample_size': 2048, 'encoder': 'byol'},
        {'dataset': 'cifar10', 'repeat': 5, **common_config, 'num_classifiers': 128, 'n_neighbors': 16,
         'batch_size': 10, 'num_experiences': 10, 'num_classes': 10, 'sample_size': 2048, 'encoder': 'byol'},
        {'dataset': 'cifar10', 'repeat': 5, **common_config, 'num_classifiers': 1024, 'n_neighbors': 32,
         'batch_size': 10, 'num_experiences': 5, 'num_classes': 10, 'sample_size': 2048, 'encoder': 'byol'},
        {'dataset': 'cifar10', 'repeat': 5, **common_config, 'num_classifiers': 1024, 'n_neighbors': 32,
         'batch_size': 10, 'num_experiences': 10, 'num_classes': 10, 'sample_size': 2048, 'encoder': 'byol'},
        # CIFAR 100
        {'dataset': 'cifar100', 'repeat': 3, **common_config, 'num_classifiers': 1024, 'n_neighbors': 32,
         'batch_size': 10, 'num_experiences': 20, 'num_classes': 100, 'sample_size': 2048, 'encoder': 'byol'},
        {'dataset': 'cifar100', 'repeat': 5, **common_config, 'num_classifiers': 1024, 'n_neighbors': 32,
         'batch_size': 10, 'num_experiences': 100, 'num_classes': 100, 'sample_size': 2048, 'encoder': 'byol'},
        {'dataset': 'cifar100', 'repeat': 5, **common_config, 'num_classifiers': 16, 'n_neighbors': 4, 'batch_size': 10,
         'num_experiences': 20, 'num_classes': 100, 'sample_size': 2048, 'encoder': 'byol'},
        {'dataset': 'cifar100', 'repeat': 5, **common_config, 'num_classifiers': 128, 'n_neighbors': 16,
         'batch_size': 10, 'num_experiences': 20, 'num_classes': 100, 'sample_size': 2048, 'encoder': 'byol'},
        {'dataset': 'cifar100', 'repeat': 5, **common_config, 'num_classifiers': 16, 'n_neighbors': 4, 'batch_size': 10,
         'num_experiences': 100, 'num_classes': 100, 'sample_size': 2048, 'encoder': 'byol'},
        {'dataset': 'cifar100', 'repeat': 5, **common_config, 'num_classifiers': 128, 'n_neighbors': 16,
         'batch_size': 10, 'num_experiences': 100, 'num_classes': 100, 'sample_size': 2048, 'encoder': 'byol'},
        # MNIST
        {'dataset': 'mnist', 'repeat': 5, **common_config, 'num_classifiers': 128, 'n_neighbors': 16, 'batch_size': 60,
         'num_experiences': 5, 'num_classes': 10, 'sample_size': 512, 'encoder': 'omniglot'},
        {'dataset': 'mnist', 'repeat': 5, **common_config, 'num_classifiers': 128, 'n_neighbors': 16, 'batch_size': 60,
         'num_experiences': 10, 'num_classes': 10, 'sample_size': 512, 'encoder': 'omniglot'},
        {'dataset': 'mnist', 'repeat': 5, **common_config, 'num_classifiers': 16, 'n_neighbors': 4, 'batch_size': 60,
         'num_experiences': 5, 'num_classes': 10, 'sample_size': 512, 'encoder': 'omniglot'},
        {'dataset': 'mnist', 'repeat': 5, **common_config, 'num_classifiers': 16, 'n_neighbors': 4, 'batch_size': 60,
         'num_experiences': 10, 'num_classes': 10, 'sample_size': 512, 'encoder': 'omniglot'},
        {'dataset': 'mnist', 'repeat': 5, **common_config, 'num_classifiers': 1024, 'n_neighbors': 32, 'batch_size': 60,
         'num_experiences': 5, 'num_classes': 10, 'sample_size': 512, 'encoder': 'omniglot'},
        {'dataset': 'mnist', 'repeat': 5, **common_config, 'num_classifiers': 1024, 'n_neighbors': 32, 'batch_size': 60,
         'num_experiences': 10, 'num_classes': 10, 'sample_size': 512, 'encoder': 'omniglot'},
    ]

    print(configs_to_run)

    ROOT_DIR = EXPERIMENTS_OUTPUT_DIR / create_default_dirname()

    for index, config in tqdm(enumerate(configs_to_run), total=len(configs_to_run), desc=f"Running experiment"):
        conf = OmegaConf.create(config)
        total_runs = conf.repeat
        for run_number in tqdm(range(total_runs), total=total_runs, desc=f"Current run"):
            logging_dir = ROOT_DIR / f'Experiment_{index}_Run_{run_number}'
            try:
                setup_and_run(config=conf, logging_dir=logging_dir)
            except Exception as exception:
                print(exception)
                print(f"Cannot train with: {config}. Skipping into ")


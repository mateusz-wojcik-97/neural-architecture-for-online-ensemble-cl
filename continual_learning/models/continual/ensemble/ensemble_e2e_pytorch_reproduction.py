import json
from collections import defaultdict
from datetime import datetime
from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics, \
    forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training import BaseStrategy
from avalanche.training.plugins import EvaluationPlugin
from omegaconf import OmegaConf
from torch.optim import Optimizer

from continual_learning.datasets.ensemble_e2e.dataset_loader import load_scenario
from continual_learning.experiments.ensemble_e2e.encoder_loader import load_encoder
from continual_learning.metrics.avalanche.accuracy import accuracy_metrics_from_strategy_state
from continual_learning.metrics.avalanche.backward_transfer import bwt_metrics_from_strategy_state
from continual_learning.metrics.avalanche.forgetting import forgetting_metrics_from_strategy_state
from continual_learning.metrics.avalanche.forward_transfer import \
    forward_transfer_metrics_from_strategy_state
from continual_learning.models.continual.ensemble.ensemble_e2e_module_reproduction import \
    EnsembleE2EModule
from continual_learning.optimizers.naive_optimizer import NaiveOptimizer
from settings import CONFIGS_DIR, EXPERIMENTS_OUTPUT_DIR


def ensemble_loss(
    vanilla_outputs: torch.Tensor,
    tanh_outputs: torch.Tensor,
    ensemble_outputs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Label to one hot
    labels_one_hot = nn.functional.one_hot(labels, num_classes=num_classes)
    # Classifier loss
    vanilla_loss = -torch.mean(torch.sum(vanilla_outputs * labels_one_hot, dim=1))
    tanh_loss = -torch.mean(torch.sum(tanh_outputs * labels_one_hot, dim=1))
    # Memory loss
    memory_loss = -torch.mean(torch.sum(ensemble_outputs * labels_one_hot, dim=1))
    # Total loss
    loss = vanilla_loss + tanh_loss + memory_loss
    return loss


class EnsembleE2E(BaseStrategy):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Any,
        **kwargs,
    ):
        super().__init__(model, optimizer, criterion, **kwargs)
        self.state = {}

    def criterion(self):
        """ Loss function. """
        return self._criterion(
            vanilla_outputs=self.state['vanilla'],
            tanh_outputs=self.state['tanh'],
            ensemble_outputs=self.state['ensemble'],
            labels=self.mb_y,
            num_classes=self.model.num_classes,
        )

    def training_epoch(self, **kwargs):
        """
        Training epoch.
        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            ensemble_out, tanh_out, vanilla_out = self.forward()
            self.mb_output = ensemble_out
            self.state = {
                'ensemble': ensemble_out,
                'tanh': tanh_out,
                'vanilla': vanilla_out,
            }
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.loss.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer.step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def eval_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)
            ensemble_out, tanh_out, vanilla_out = self.forward()
            self.mb_output = ensemble_out
            self.state = {
                'ensemble': ensemble_out,
                'tanh': tanh_out,
                'vanilla': vanilla_out,
            }
            self._after_eval_forward(**kwargs)
            self.loss = self.criterion()

            self._after_eval_iteration(**kwargs)

    def make_optimizer(self):
        # Optimizer is stateless so there is no need to reset
        pass


def create_default_dirname(prefix: str = 'experiment') -> str:
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    return f"{prefix}-{current_time}"


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")  # ".*does not have many workers.*"

    CONFIG_PATH = CONFIGS_DIR / 'ensemble_e2e' / 'config.yaml'
    DATASET = 'mnist'

    # Config
    config = OmegaConf.load(CONFIG_PATH)[DATASET]

    print(config)

    # Scenario (dataset)
    scenario = load_scenario(dataset_name=DATASET, config=config)

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
    experience_eval = True
    eval_plugin = EvaluationPlugin(
        # accuracy_metrics(minibatch=False, epoch=False, experience=experience_eval, stream=True),
        accuracy_metrics_from_strategy_state(state_keys=state_keys, minibatch=False, epoch=False, experience=experience_eval, stream=True),
        loss_metrics(minibatch=False, epoch=False, experience=experience_eval, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=experience_eval, stream=True),
        forgetting_metrics_from_strategy_state(state_keys=state_keys, experience=experience_eval, stream=True),
        bwt_metrics_from_strategy_state(state_keys=state_keys, experience=experience_eval, stream=True),
        # forward_transfer_metrics_from_strategy_state(state_keys=state_keys, experience=experience_eval, stream=True),
        # confusion_matrix_metrics(num_classes=num_classes, save_image=False, stream=True),
        loggers=[InteractiveLogger()],
        benchmark=scenario,
        strict_checks=False
    )

    # Strategy
    strategy = EnsembleE2E(
        model=classifier,
        optimizer=optimizer,
        criterion=ensemble_loss,
        evaluator=eval_plugin,
        # Kwargs
        train_mb_size=config.batch_size,
        train_epochs=config.train_epochs,
        eval_mb_size=config.batch_size,
        device=config.device,
        eval_every=-1,
    )

    results = defaultdict(list)

    log_folder = EXPERIMENTS_OUTPUT_DIR / create_default_dirname()
    log_folder.mkdir(parents=True, exist_ok=True)
    print(f"Logging into folder: {log_folder}")

    train_stream = scenario.train_stream
    test_stream = scenario.test_stream
    for train_task in train_stream:
        strategy.train(train_task, num_workers=6)
        eval_result = strategy.eval(test_stream)
        results['data'].append(eval_result)

        with open(log_folder / 'experiment_log.json', 'w') as fp:
            json.dump(results, fp)

    print('======================== Results ========================')
    print(results)

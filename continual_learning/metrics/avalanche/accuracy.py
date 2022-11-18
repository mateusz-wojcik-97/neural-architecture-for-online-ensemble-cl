################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import List

from avalanche.evaluation import PluginMetric, GenericPluginMetric
from avalanche.evaluation.metric_utils import phase_and_task
from avalanche.evaluation.metrics import Accuracy


class AccuracyPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all accuracies plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, from_state_key: str):
        self._accuracy = Accuracy()
        self.from_state_key = from_state_key
        super(AccuracyPluginMetric, self).__init__(
            self._accuracy, reset_at=reset_at, emit_at=emit_at,
            mode=mode)

    def reset(self, strategy=None) -> None:
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])

    def result(self, strategy=None) -> float:
        if self._emit_at == 'stream' or strategy is None:
            return self._metric.result()
        else:
            return self._metric.result(phase_and_task(strategy)[1])

    def update(self, strategy):
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
        self._accuracy.update(strategy.state[self.from_state_key], strategy.mb_y, task_labels)


class MinibatchAccuracy(AccuracyPluginMetric):
    """
    The minibatch plugin accuracy metric.
    This metric only works at training time.

    This metric computes the average accuracy over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochAccuracy` instead.
    """

    def __init__(self, from_state_key: str):
        """
        Creates an instance of the MinibatchAccuracy metric.
        """
        super(MinibatchAccuracy, self).__init__(
            reset_at='iteration', emit_at='iteration', mode='train', from_state_key=from_state_key)

    def __str__(self):
        return f"Top1_Acc_MB_{self.from_state_key}"


class EpochAccuracy(AccuracyPluginMetric):
    """
    The average accuracy over a single training epoch.
    This plugin metric only works at training time.

    The accuracy will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self, from_state_key: str):
        """
        Creates an instance of the EpochAccuracy metric.
        """

        super(EpochAccuracy, self).__init__(
            reset_at='epoch', emit_at='epoch', mode='train', from_state_key=from_state_key)

    def __str__(self):
        return f"Top1_Acc_Epoch_{self.from_state_key}"


class RunningEpochAccuracy(AccuracyPluginMetric):
    """
    The average accuracy across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the accuracy averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self, from_state_key: str):
        """
        Creates an instance of the RunningEpochAccuracy metric.
        """

        super(RunningEpochAccuracy, self).__init__(
            reset_at='epoch', emit_at='iteration', mode='train', from_state_key=from_state_key)

    def __str__(self):
        return f"Top1_RunningAcc_Epoch_{self.from_state_key}"


class ExperienceAccuracy(AccuracyPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self, from_state_key: str):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceAccuracy, self).__init__(
            reset_at='experience', emit_at='experience', mode='eval', from_state_key=from_state_key)

    def __str__(self):
        return f"Top1_Acc_Exp_{self.from_state_key}"


class StreamAccuracy(AccuracyPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average accuracy over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self, from_state_key: str):
        """
        Creates an instance of StreamAccuracy metric
        """
        super(StreamAccuracy, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval', from_state_key=from_state_key)

    def __str__(self):
        return f"Top1_Acc_Stream_{self.from_state_key}"


class TrainedExperienceAccuracy(AccuracyPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    accuracy for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self, from_state_key: str):
        """
        Creates an instance of TrainedExperienceAccuracy metric by first
        constructing AccuracyPluginMetric
        """
        super(TrainedExperienceAccuracy, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval', from_state_key=from_state_key)
        self._current_experience = 0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience
        AccuracyPluginMetric.reset(self, strategy)
        return AccuracyPluginMetric.after_training_exp(self, strategy)

    def update(self, strategy):
        """
        Only update the accuracy with results from experiences that have been
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            AccuracyPluginMetric.update(self, strategy)

    def __str__(self):
        return f"Accuracy_On_Trained_Experiences_{self.from_state_key}"


def accuracy_metrics_from_strategy_state(
    *,
    state_keys: List[str],
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
    trained_experience=False,
) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch accuracy at training time.
    :param epoch: If True, will return a metric able to log
        the epoch accuracy at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch accuracy at training time.
    :param experience: If True, will return a metric able to log
        the accuracy on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the accuracy averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation accuracy only for experiences that the
        model has been trained on

    :return: A list of plugin metrics.
    """

    metrics = []
    for state_key in state_keys:
        if minibatch:
            metrics.append(MinibatchAccuracy(from_state_key=state_key))

        if epoch:
            metrics.append(EpochAccuracy(from_state_key=state_key))

        if epoch_running:
            metrics.append(RunningEpochAccuracy(from_state_key=state_key))

        if experience:
            metrics.append(ExperienceAccuracy(from_state_key=state_key))

        if stream:
            metrics.append(StreamAccuracy(from_state_key=state_key))

        if trained_experience:
            metrics.append(TrainedExperienceAccuracy(from_state_key=state_key))

    return metrics


__all__ = [
    'Accuracy',
    'MinibatchAccuracy',
    'EpochAccuracy',
    'RunningEpochAccuracy',
    'ExperienceAccuracy',
    'StreamAccuracy',
    'TrainedExperienceAccuracy',
    'accuracy_metrics_from_strategy_state'
]

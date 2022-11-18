from typing import List, Dict, Union

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics import Forgetting

from continual_learning.metrics.avalanche.forgetting import ExperienceForgetting, StreamForgetting


def forgetting_to_bwt(f):
    """
    Convert forgetting to backward transfer.
    BWT = -1 * forgetting
    """
    if f is None:
        return f
    if isinstance(f, dict):
        bwt = {k: -1 * v for k, v in f.items()}
    elif isinstance(f, float):
        bwt = -1 * f
    else:
        raise ValueError("Forgetting data type not recognized when converting"
                         "to backward transfer.")
    return bwt


class BWT(Forgetting):
    """
    The standalone Backward Transfer metric.
    This metric returns the backward transfer relative to a specific key.
    Alternatively, this metric returns a dict in which each key is associated
    to the backward transfer.
    Backward transfer is computed as the difference between the last value
    recorded for a specific key and the first value recorded for that key.
    The value associated to a key can be update with the `update` method.

    At initialization, this metric returns an empty dictionary.
    """

    def result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        Backward Transfer is returned only for keys encountered twice.
        Backward Transfer is the negative forgetting.

        :param k: the key for which returning backward transfer. If k has not
            updated at least twice it returns None. If k is None,
            backward transfer will be returned for all keys encountered at
            least twice.

        :return: the difference between the last value encountered for k
            and its first value, if k is not None.
            It returns None if k has not been updated
            at least twice. If k is None, returns a dictionary
            containing keys whose value has been updated at least twice. The
            associated value is the difference between the last and first
            value recorded for that key.
        """

        forgetting = super().result(k)
        bwt = forgetting_to_bwt(forgetting)
        return bwt


class ExperienceBWT(ExperienceForgetting):
    """
    The Experience Backward Transfer metric.

    This plugin metric, computed separately for each experience,
    is the difference between the last accuracy result obtained on a certain
    experience and the accuracy result obtained when first training on that
    experience.

    This metric is computed during the eval phase only.
    """

    def result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        See `Forgetting` documentation for more detailed information.

        k: optional key from which compute forgetting.
        """
        forgetting = super().result(k)
        return forgetting_to_bwt(forgetting)

    def __str__(self):
        return f"ExperienceBWT_{self.from_state_key}"


class StreamBWT(StreamForgetting):
    """
    The StreamBWT metric, emitting the average BWT across all experiences
    encountered during training.

    This plugin metric, computed over all observed experiences during training,
    is the average over the difference between the last accuracy result
    obtained on an experience and the accuracy result obtained when first
    training on that experience.

    This metric is computed during the eval phase only.
    """

    def exp_result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        Result for experience defined by a key.
        See `BWT` documentation for more detailed information.

        k: optional key from which compute backward transfer.
        """
        forgetting = super().exp_result(k)
        return forgetting_to_bwt(forgetting)

    def __str__(self):
        return f"StreamBWT_{self.from_state_key}"


def bwt_metrics_from_strategy_state(*, state_keys: List[str], experience=False, stream=False) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param experience: If True, will return a metric able to log
        the backward transfer on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the backward transfer averaged over the evaluation stream experiences
        which have been observed during training.
    :return: A list of plugin metrics.
    """

    metrics = []
    for state_key in state_keys:
        if experience:
            metrics.append(ExperienceBWT(from_state_key=state_key))

        if stream:
            metrics.append(StreamBWT(from_state_key=state_key))

    return metrics


__all__ = [
    'bwt_metrics_from_strategy_state'
]

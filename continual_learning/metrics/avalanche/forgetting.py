from typing import List, Dict, Union

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.evaluation.metric_utils import get_metric_name, phase_and_task, stream_type
from avalanche.evaluation.metrics import Forgetting, Accuracy, Mean


class GenericExperienceForgetting(PluginMetric[Dict[int, float]]):
    """
    The GenericExperienceForgetting metric, describing the change in
    a metric detected for a certain experience. The user should
    subclass this and provide the desired metric.

    In particular, the user should override:
    * __init__ by calling `super` and instantiating the `self.current_metric`
    property as a valid avalanche metric
    * `metric_update`, to update `current_metric`
    * `metric_result` to get the result from `current_metric`.
    * `__str__` to define the experience forgetting  name.

    This plugin metric, computed separately for each experience,
    is the difference between the metric result obtained after
    first training on a experience and the metric result obtained
    on the same experience at the end of successive experiences.

    This metric is computed during the eval phase only.
    """

    def __init__(self, from_state_key: str):
        """
        Creates an instance of the GenericExperienceForgetting metric.
        """

        super().__init__()

        self.from_state_key = from_state_key

        self.forgetting = Forgetting()
        """
        The general metric to compute forgetting
        """

        self._current_metric = None
        """
        The metric the user should override
        """

        self.eval_exp_id = None
        """
        The current evaluation experience id
        """

        self.train_exp_id = None
        """
        The last encountered training experience id
        """

    def reset(self) -> None:
        """
        Resets the metric.

        Beware that this will also reset the initial metric of each
        experience!

        :return: None.
        """
        self.forgetting.reset()

    def reset_last(self) -> None:
        """
        Resets the last metric value.

        This will preserve the initial metric value of each experience.
        To be used at the beginning of each eval experience.

        :return: None.
        """
        self.forgetting.reset_last()

    def update(self, k, v, initial=False):
        """
        Update forgetting metric.
        See `Forgetting` for more detailed information.

        :param k: key to update
        :param v: value associated to k
        :param initial: update initial value. If False, update
            last value.
        """
        self.forgetting.update(k, v, initial=initial)

    def result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        See `Forgetting` documentation for more detailed information.

        k: optional key from which compute forgetting.
        """
        return self.forgetting.result(k=k)

    def before_training_exp(self, strategy: 'BaseStrategy') -> None:
        self.train_exp_id = strategy.experience.current_experience

    def before_eval(self, strategy) -> None:
        self.reset_last()

    def before_eval_exp(self, strategy: 'BaseStrategy') -> None:
        self._current_metric.reset()

    def after_eval_iteration(self, strategy: 'BaseStrategy') -> None:
        super().after_eval_iteration(strategy)
        self.eval_exp_id = strategy.experience.current_experience
        self.metric_update(strategy)

    def after_eval_exp(self, strategy: 'BaseStrategy') \
            -> MetricResult:
        # update experience on which training just ended
        if self.train_exp_id == self.eval_exp_id:
            self.update(self.eval_exp_id,
                        self.metric_result(strategy),
                        initial=True)
        else:
            # update other experiences
            # if experience has not been encountered in training
            # its value will not be considered in forgetting
            self.update(self.eval_exp_id,
                        self.metric_result(strategy))

        return self._package_result(strategy)

    def _package_result(self, strategy: 'BaseStrategy') \
            -> MetricResult:
        # this checks if the evaluation experience has been
        # already encountered at training time
        # before the last training.
        # If not, forgetting should not be returned.
        forgetting = self.result(k=self.eval_exp_id)
        if forgetting is not None:
            metric_name = get_metric_name(self, strategy, add_experience=True)
            plot_x_position = strategy.clock.train_iterations

            metric_values = [MetricValue(
                self, metric_name, forgetting, plot_x_position)]
            return metric_values

    def metric_update(self, strategy):
        raise NotImplementedError

    def metric_result(self, strategy):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class ExperienceForgetting(GenericExperienceForgetting):
    """
    The ExperienceForgetting metric, describing the accuracy loss
    detected for a certain experience.

    This plugin metric, computed separately for each experience,
    is the difference between the accuracy result obtained after
    first training on a experience and the accuracy result obtained
    on the same experience at the end of successive experiences.

    This metric is computed during the eval phase only.
    """

    def __init__(self, from_state_key: str):
        """
        Creates an instance of the ExperienceForgetting metric.
        """

        super().__init__(from_state_key=from_state_key)

        self._current_metric = Accuracy()
        """
        The average accuracy over the current evaluation experience
        """

    def metric_update(self, strategy):
        self._current_metric.update(strategy.mb_y,
                                    strategy.state[self.from_state_key], 0)

    def metric_result(self, strategy):
        return self._current_metric.result(0)[0]

    def __str__(self):
        return f"ExperienceForgetting_{self.from_state_key}"


class GenericStreamForgetting(GenericExperienceForgetting):
    """
    The GenericStreamForgetting metric, describing the average evaluation
    change in the desired metric detected over all experiences observed
    during training.

    In particular, the user should override:
    * __init__ by calling `super` and instantiating the `self.current_metric`
    property as a valid avalanche metric
    * `metric_update`, to update `current_metric`
    * `metric_result` to get the result from `current_metric`.
    * `__str__` to define the experience forgetting  name.

    This plugin metric, computed over all observed experiences during training,
    is the average over the difference between the metric result obtained
    after first training on a experience and the metric result obtained
    on the same experience at the end of successive experiences.

    This metric is computed during the eval phase only.
    """

    def __init__(self, from_state_key: str):
        """
        Creates an instance of the GenericStreamForgetting metric.
        """

        super().__init__(from_state_key)

        self.stream_forgetting = Mean()
        """
        The average forgetting over all experiences
        """

    def reset(self) -> None:
        """
        Resets the forgetting metrics.

        Beware that this will also reset the initial metric value of each
        experience!

        :return: None.
        """
        super().reset()
        self.stream_forgetting.reset()

    def exp_update(self, k, v, initial=False):
        """
        Update forgetting metric.
        See `Forgetting` for more detailed information.

        :param k: key to update
        :param v: value associated to k
        :param initial: update initial value. If False, update
            last value.
        """
        super().update(k, v, initial=initial)

    def exp_result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        Result for experience defined by a key.
        See `Forgetting` documentation for more detailed information.

        k: optional key from which compute forgetting.
        """
        return super().result(k)

    def result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        The average forgetting over all experience.

        k: optional key from which compute forgetting.
        """
        return self.stream_forgetting.result()

    def before_eval(self, strategy) -> None:
        super().before_eval(strategy)
        self.stream_forgetting.reset()

    def after_eval_exp(self, strategy: 'BaseStrategy') -> None:
        # update experience on which training just ended
        if self.train_exp_id == self.eval_exp_id:
            self.exp_update(self.eval_exp_id,
                            self.metric_result(strategy),
                            initial=True)
        else:
            # update other experiences
            # if experience has not been encountered in training
            # its value will not be considered in forgetting
            self.exp_update(self.eval_exp_id,
                            self.metric_result(strategy))

        # this checks if the evaluation experience has been
        # already encountered at training time
        # before the last training.
        # If not, forgetting should not be returned.
        exp_forgetting = self.exp_result(k=self.eval_exp_id)
        if exp_forgetting is not None:
            self.stream_forgetting.update(exp_forgetting, weight=1)

    def after_eval(self, strategy: 'BaseStrategy') -> \
            'MetricResult':
        return self._package_result(strategy)

    def _package_result(self, strategy: 'BaseStrategy') -> \
            MetricResult:
        metric_value = self.result()

        phase_name, _ = phase_and_task(strategy)
        stream = stream_type(strategy.experience)
        metric_name = '{}/{}_phase/{}_stream' \
            .format(str(self),
                    phase_name,
                    stream)
        plot_x_position = strategy.clock.train_iterations

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def metric_update(self, strategy):
        raise NotImplementedError

    def metric_result(self, strategy):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class StreamForgetting(GenericStreamForgetting):
    """
    The StreamForgetting metric, describing the average evaluation accuracy loss
    detected over all experiences observed during training.

    This plugin metric, computed over all observed experiences during training,
    is the average over the difference between the accuracy result obtained
    after first training on a experience and the accuracy result obtained
    on the same experience at the end of successive experiences.

    This metric is computed during the eval phase only.
    """

    def __init__(self, from_state_key: str):
        """
        Creates an instance of the StreamForgetting metric.
        """

        super().__init__(from_state_key)

        self._current_metric = Accuracy()
        """
        The average accuracy over the current evaluation experience
        """

    def metric_update(self, strategy):
        self._current_metric.update(strategy.mb_y,
                                    strategy.state[self.from_state_key], 0)

    def metric_result(self, strategy):
        return self._current_metric.result(0)[0]

    def __str__(self):
        return f"StreamForgetting_{self.from_state_key}"


def forgetting_metrics_from_strategy_state(*, state_keys: List[str], experience=False, stream=False) \
        -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param experience: If True, will return a metric able to log
        the forgetting on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the forgetting averaged over the evaluation stream experiences,
        which have been observed during training.

    :return: A list of plugin metrics.
    """

    metrics = []
    for state_key in state_keys:
        if experience:
            metrics.append(ExperienceForgetting(from_state_key=state_key))

        if stream:
            metrics.append(StreamForgetting(from_state_key=state_key))

    return metrics


__all__ = [
    'forgetting_metrics_from_strategy_state'
]

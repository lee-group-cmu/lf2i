from typing import Union, Dict, Any, List
from abc import ABC, abstractmethod

from lf2i.test_statistics._estimators import ESTIMATORS


class TestStatistic(ABC):
    """Base class for test statistics. This is a template from which every test statistic should inherit.

    Parameters
    ----------
    acceptance_region : str
        Whether the acceptance region for the corresponding test is defined to be on the right or on the left of the critical value. 
        Must be either `left` or `right`.
    estimator_kwargs: Dict
        Hyperparameters and settings for the estimator, by default {}.
    """
    
    def __init__(
        self,
        acceptance_region: str,
    ) -> None:
        self.acceptance_region = acceptance_region
        self._estimator_trained = dict()
    
    def _choose_estimator(
        self, 
        estimator: Union[str, Any],
        estimator_kwargs: Dict,
        estimand_name: str
    ) -> Any:
        if isinstance(estimator, str):
            self._estimator_trained[estimand_name] = False
            if estimator not in ESTIMATORS:
                raise ValueError(f'Invalid estimator name. Available: {list(ESTIMATORS.keys())}; got {estimator}')
            return ESTIMATORS[estimator](**estimator_kwargs)
        else:
            # just flag it as trained
            self._estimator_trained[estimand_name] = True
            return estimator
    
    def _check_is_trained(
        self
    ) -> List[bool]:
        return all([is_trained for _, is_trained in self._estimator_trained.items()])

    @abstractmethod
    def estimate(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

from typing import Union
from abc import ABC, abstractmethod

from lf2i.test_statistics._estimators import ESTIMATORS


class TestStatistic(ABC):
    """Base class for test statistics. This is a template from which every test statistic should inherit.

    Parameters
    ----------
    estimator : Union[str, object]
        Model used to estimate the building blocks of the test statistic. 
        If `object`, it is expected to be trained. If `str`, it will need to be trained by calling `self.estimate`.
    acceptance_region : str
        Whether the acceptance region for the corresponding test is defined to be on the right or on the left of the critical value. 
        Must be either `left` or `right`.
    """
    
    def __init__(
        self,
        estimator: Union[str, object],  # not sure what would be the type of a general estimator
        acceptance_region: str
    ) -> None:
        self.estimator = self._choose_estimator(estimator, 'estimand')  # leave it general for now
        self.acceptance_region = acceptance_region
        
        self._estimator_trained = dict()
    
    def _choose_estimator(
        self, 
        estimator: Union[str, object],
        estimand_name: str
    ) -> object:
        if isinstance(estimator, str):
            self._estimator_trained[estimand_name] = False
            if estimator not in ESTIMATORS:
                raise ValueError(f'Invalid estimator name. Available: {list(ESTIMATORS.keys())}; got {estimator}')
            return ESTIMATORS[estimator]
        else:
            # just flag it as trained
            self._estimator_trained[estimand_name] = True
            return estimator
    
    def _check_is_trained(
        self
    ) -> None:
        if not all([is_trained for _, is_trained in self._estimator_trained.items()]):
            raise RuntimeError("Not all needed estimators are trained. Check self._estimator_trained")

    @abstractmethod
    def estimate(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

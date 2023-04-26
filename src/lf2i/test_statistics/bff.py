from typing import Union, Any, Dict

import numpy as np
import torch

from lf2i.test_statistics._base import TestStatistic
from lf2i.utils.likelihood_inputs import preprocess_odds_estimation, preprocess_odds_cv, preprocess_odds_cs


class BFF(TestStatistic):
    
    def __init__(
        self,
        estimator: Union[str, Any],
        hp_test_type: str,
        param_dim: int,
        data_sample_size: int,
        estimator_kwargs: Dict = {}
    ) -> None:
        super().__init__(acceptance_region='right', estimation_method='likelihood')

        self.hp_test_type = hp_test_type
        self.param_dim = param_dim
        self.data_sample_size = data_sample_size
        self.estimator = self._choose_estimator(estimator, estimator_kwargs, 'odds')

    def estimate(
        self,
        labels: Union[np.ndarray, torch.Tensor], 
        parameters: Union[np.ndarray, torch.Tensor], 
        samples: Union[np.ndarray, torch.Tensor]
    ) -> None:
        labels, params_samples = preprocess_odds_estimation(labels, parameters, samples, self.param_dim, self.estimator)
        self.estimator.fit(X=params_samples, y=parameters)
        self._estimator_trained['odds'] = True
    
    def _compute_for_critical_values(
        self,
        parameters: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        params_samples = preprocess_odds_cv(parameters, samples, self.param_dim, self.data_sample_size, self.estimator.model)  # TODO: ??
        if (self.hp_test_type == 'simple') and (self.data_sample_size == 1):
            # in this case BFF denominator == 1 and numerator is only odds. 
            # N.B.: Technically we also need G to be the marginal of F_\theta
            probabilities = self.estimator.predict_proba(X=params_samples)
            return probabilities[:, 1] / probabilities[:, 0]
        else:
            raise NotImplementedError
    
    def _compute_for_confidence_sets(
        self, 
        parameter_grid: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        params_samples = preprocess_odds_cs(parameter_grid, samples, self.param_dim, self.data_sample_size, self.estimator.model)
        if (self.hp_test_type == 'simple') and (self.data_sample_size == 1):
            # in this case BFF denominator == 1 and numerator is only odds. 
            # N.B.: Technically we also need G to be the marginal of F_\theta
            probabilities = self.estimator.predict_proba(X=params_samples)
            return (probabilities[:, 1] / probabilities[:, 0]).reshape(-1, parameter_grid.reshape(-1, self.param_dim).shape[0])
        else:
            raise NotImplementedError

    def _compute_for_diagnostics(
        self,
        parameters: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        return self._compute_for_critical_values(parameters, samples)

    def evaluate(
        self,
        parameters: Union[np.ndarray, torch.Tensor],
        samples:  Union[np.ndarray, torch.Tensor],
        mode: str
    ) -> np.ndarray:
        if mode == 'critical_values':
            return self._compute_for_critical_values(parameters, samples)
        elif mode == 'confidence_sets':
            return self._compute_for_confidence_sets(parameters, samples)
        elif mode == 'diagnostics':
            return self._compute_for_diagnostics(parameters, samples)
        else:
            raise ValueError(f"Only `critical_values`, `confidence_sets`, and `diagnostics` are supported, got {mode}")

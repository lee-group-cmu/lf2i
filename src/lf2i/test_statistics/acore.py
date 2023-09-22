from typing import Union, Any, Dict, Optional, List

import numpy as np
import torch
from scipy.optimize import minimize

from lf2i.test_statistics._base import TestStatistic
from lf2i.utils.likelihood_inputs import preprocess_odds_estimation, preprocess_for_odds_cv


class ACORE(TestStatistic):
    
    def __init__(
        self,
        estimator: Union[str, Any],
        poi_dim: int,
        nuisance_dim: int,
        data_sample_size: int,
        data_dim: int,
        estimator_kwargs: Dict = {},
        n_jobs: int = 1
    ) -> None:
        
        super().__init__(acceptance_region='right', estimation_method='likelihood')

        self.poi_dim = poi_dim
        self.nuisance_dim = nuisance_dim
        self.param_dim = poi_dim + nuisance_dim
        self.data_sample_size = data_sample_size
        self.data_dim = data_dim
        self.estimator = self._choose_estimator(estimator, estimator_kwargs, 'odds')
        self.n_jobs = n_jobs
    
    def estimate(
        self,
        labels: Union[np.ndarray, torch.Tensor], 
        parameters: Union[np.ndarray, torch.Tensor], 
        samples: Union[np.ndarray, torch.Tensor],
    ) -> None:
        labels, params_samples = preprocess_odds_estimation(labels, parameters, samples, self.param_dim, self.estimator)
        self.estimator.fit(X=params_samples, y=labels)
        self._estimator_trained['odds'] = True

    def evaluate(
        self,
        parameters: Union[np.ndarray, torch.Tensor],
        samples:  Union[np.ndarray, torch.Tensor],
        mode: str,
        param_space_bounds: Optional[List[List[float]]] = None,
        hybrid: bool = False
    ) -> np.ndarray:
        if mode == 'critical_values':
            return self._compute_for_critical_values(parameters, samples, param_space_bounds, hybrid)
        elif mode == 'confidence_sets':
            return self._compute_for_confidence_sets(parameters, samples, hybrid)
        elif mode == 'diagnostics':
            return self._compute_for_diagnostics(parameters, samples, param_space_bounds, hybrid)
        else:
            raise ValueError(f"Only `critical_values`, `confidence_sets`, and `diagnostics` are supported, got {mode}")
    
    def _odds_ratio(
        self,
        probs_theta0: Union[np.ndarray, torch.Tensor],
        probs_theta1: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        pass

    def _optimize_log_odds_ratio(
        self,
        probs_theta0: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        pass

    def _compute_for_critical_values(
        self,
        parameters: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor, None],
        param_space_bounds: Optional[List[List[float]]] = None,
        hybrid: bool = False,
        simulator: Optional[Any] = None
    ) -> None:
        # NOTE: this only considers simple null hypothesis with respect to the POI, which is what we need for confidence sets
        parameters, samples, params_samples = preprocess_for_odds_cv(parameters, samples, self.param_dim, self.data_sample_size, self.data_dim, self.estimator)
        if self.nuisance_dim == 0:
            return self._optimize_log_odds_ratio(self.estimator.predict_proba(X=params_samples))
        else:
            raise NotImplementedError
    
    def _compute_for_confidence_sets(
        self, 
        parameter_grid: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor],
        param_space_bounds: List[List[float]]
    ) -> np.ndarray:
        pass

    def _compute_for_diagnostics(
        self,
        parameters: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor],
        param_space_bounds: List[List[float]]
    ) -> np.ndarray:
        return self._compute_for_critical_values(parameters, samples, param_space_bounds)

from typing import Union, Any, Dict, List
from tqdm import tqdm

import numpy as np
from scipy import integrate
import torch

from lf2i.test_statistics._base import TestStatistic
from lf2i.utils.likelihood_inputs import preprocess_odds_estimation, preprocess_for_odds_cv, preprocess_for_odds_cs


class BFF(TestStatistic):
    
    def __init__(
        self,
        estimator: Union[str, Any],
        task: str,  # either 'inference' or 'anomaly_detection'
        poi_dim: int,
        nuisance_dim: int,
        data_sample_size: int,
        data_dim: int,
        estimator_kwargs: Dict = {}
    ) -> None:
        super().__init__(acceptance_region='right', estimation_method='likelihood')

        self.task = task
        self.poi_dim = poi_dim
        self.nuisance_dim = nuisance_dim
        self.param_dim = poi_dim + nuisance_dim
        self.data_sample_size = data_sample_size
        self.data_dim = data_dim
        self.estimator = self._choose_estimator(estimator, estimator_kwargs, 'odds')

    def estimate(
        self,
        labels: Union[np.ndarray, torch.Tensor], 
        parameters: Union[np.ndarray, torch.Tensor], 
        samples: Union[np.ndarray, torch.Tensor],
    ) -> None:
        labels, params_samples = preprocess_odds_estimation(labels, parameters, samples, self.param_dim, self.estimator)
        self.estimator.fit(X=params_samples, y=labels)
        self._estimator_trained['odds'] = True

    def _odds(
        self,
        probabilities: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().detach().numpy()
        return np.prod((probabilities[:, 1] / probabilities[:, 0]).reshape(-1, self.data_sample_size), axis=1)

    def _integrate_odds(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        fixed_poi: Union[np.ndarray, torch.Tensor],  # needed only if integrating solely over nuisances; otherwise empty array
        integration_bounds: List[List[float]]
    ) -> float:
        return integrate.nquad(
            func=lambda *params: self._odds(self.estimator.predict_proba(
                X=np.hstack((
                    np.repeat(np.array([*fixed_poi, *params]).reshape(-1, self.param_dim), repeats=self.data_sample_size), 
                    samples
                ))
            )),
            ranges=integration_bounds
        )[0]  # return only the result of the integration
        
    def _compute_for_critical_values(
        self,
        parameters: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor],
        param_space_bounds: List[List[float]]
    ) -> np.ndarray:
        
        # TODO: 
        # 1) DONE --> restructure to avoid redundancies, if possible; 
        # 2) fix usage of param_space_bounds, allow integration over any prior;
        # 3) check if better to do computations with log (see last appendix in lf2i paper)
        # 4) optimize (minimize calls to predict_proba, vectorize, numba, parallelize); 
        # 5) check and fix consistency of types for arrays-estimators
        parameters, samples, params_samples = preprocess_for_odds_cv(parameters, samples, self.param_dim, self.data_sample_size, self.data_dim, self.estimator)
        if self.task == 'inference':
            if self.nuisance_dim == 0:
                if self.data_sample_size == 1:
                    # TODO: technically we also need G to be the marginal of F_\theta, and proportion of Y=1 to be 0.5
                    # in this case BFF denominator == 1 and numerator is only odds. 
                    return self._odds(self.estimator.predict_proba(X=params_samples))
                else:
                    numerator = self._odds(self.estimator.predict_proba(X=params_samples))
                    denominator = np.array([
                        self._integrate_odds(samples=samples[i, :, :], fixed_poi=np.empty(0), integration_bounds=param_space_bounds[:self.poi_dim]) 
                        for i in tqdm(range(samples.shape[0]))
                    ])
                    return numerator / denominator
            else:
                return np.array([
                    self._integrate_odds(samples=samples[i, :, :], fixed_poi=parameters[i, :self.poi_dim], integration_bounds=param_space_bounds[-self.nuisance_dim:]) / 
                    self._integrate_odds(samples=samples[i, :, :], fixed_poi=np.empty(0), integration_bounds=param_space_bounds)
                    for i in tqdm(range(samples.shape[0]))
                ])
        elif self.task == 'anomaly_detection':
            raise NotImplementedError
    
    def _compute_for_confidence_sets(
        self, 
        parameter_grid: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor],
        param_space_bounds: List[List[float]]
    ) -> np.ndarray:
        # TODO: can we call _compute_for_critical_values instead of rewriting much similar code?

        parameter_grid, samples, param_grid_samples = preprocess_for_odds_cs(parameter_grid, samples, self.poi_dim, self.data_sample_size, self.data_dim, self.estimator)
        if self.nuisance_dim == 0:
            if self.data_sample_size == 1:
                # TODO: see same point in _compute_for_critical_values
                return self._odds(self.estimator.predict_proba(X=param_grid_samples)).reshape(samples.shape[0], parameter_grid.shape[0])
            else:
                numerator = self._odds(self.estimator.predict_proba(X=param_grid_samples)).reshape(samples.shape[0], parameter_grid.shape[0])
                # denominator is the same regardless of parameter grid value
                denominator = np.array([
                    self._integrate_odds(samples=samples[i, :, :], fixed_poi=np.empty(0), integration_bounds=param_space_bounds[:self.poi_dim]) 
                    for i in tqdm(range(samples.shape[0]))
                ]).reshape(-1, 1)
                return numerator / denominator  # automatic broadcasting along dimension 1
        else:
            out = np.empty(shape=(samples.shape[0], parameter_grid.shape[0]))
            for i in tqdm(range(samples.shape[0])):
                # denominator is the same regardless of parameter grid value
                denominator = self._integrate_odds(samples=samples[i, :, :], fixed_poi=np.empty(0), integration_bounds=param_space_bounds)
                for j in range(parameter_grid.shape[0]):
                    numerator = self._integrate_odds(samples=samples[i, :, :], fixed_poi=parameter_grid[j, :], integration_bounds=param_space_bounds[-self.nuisance_dim:])
                    out[i][j] = numerator / denominator
            return out

    def _compute_for_diagnostics(
        self,
        parameters: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor],
        param_space_bounds: List[List[float]]
    ) -> np.ndarray:
        return self._compute_for_critical_values(parameters, samples, param_space_bounds)

    def evaluate(
        self,
        parameters: Union[np.ndarray, torch.Tensor],
        samples:  Union[np.ndarray, torch.Tensor],
        mode: str,
        param_space_bounds: List[List[float]]
    ) -> np.ndarray:
        if mode == 'critical_values':
            return self._compute_for_critical_values(parameters, samples, param_space_bounds)
        elif mode == 'confidence_sets':
            return self._compute_for_confidence_sets(parameters, samples)
        elif mode == 'diagnostics':
            return self._compute_for_diagnostics(parameters, samples, param_space_bounds)
        else:
            raise ValueError(f"Only `critical_values`, `confidence_sets`, and `diagnostics` are supported, got {mode}")

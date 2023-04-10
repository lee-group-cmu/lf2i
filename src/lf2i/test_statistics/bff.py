from typing import Union, Any, Dict, List

from tqdm import tqdm
import numpy as np
from scipy import integrate
import torch

from lf2i.test_statistics._base import TestStatistic
from lf2i.utils.likelihood_inputs import preprocess_odds_estimation, preprocess_odds_cv, preprocess_odds_cs


class BFF(TestStatistic):
    
    def __init__(
        self,
        estimator: Union[str, Any],
        null_hp_test_type: str,
        poi_dim: int,
        nuisance_dim: int,
        data_sample_size: int,
        estimator_kwargs: Dict = {}
    ) -> None:
        super().__init__(acceptance_region='right')

        self.null_hp_test_type = null_hp_test_type
        self.poi_dim = poi_dim
        self.nuisance_dim = nuisance_dim
        self.param_dim = poi_dim + nuisance_dim
        self.data_sample_size = data_sample_size
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
        
    def _compute_for_critical_values(
        self,
        parameters: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor],
        param_space_bounds: List[List[float]]
    ) -> np.ndarray:
        
        def odds(probabilities: np.ndarray) -> np.ndarray:
            return np.prod((probabilities[:, 1] / probabilities[:, 0]).reshape(-1, self.data_sample_size), axis=1)

        # TODO: 
        # 1) restructure to avoid redundancies, if possible; 
        # 2) fix usage of param_space_bounds, allow integration over any prior;
        # 3) check if better to do computations with log (see last appendix in lf2i paper)
        # 4) optimize (minimize calls to predict_proba, vectorize, numba, parallelize); 
        # 5) check and fix consistency of types for arrays-estimators
        params_samples = preprocess_odds_cv(parameters, samples, self.param_dim, self.data_sample_size, self.estimator)
        if self.null_hp_test_type == 'simple':
            if self.nuisance_dim == 0:
                probabilities = self.estimator.predict_proba(X=params_samples).astype(np.double)
                if self.data_sample_size == 1:
                    # TODO: technically we also need G to be the marginal of F_\theta, and proportion of Y=1 to be 0.5
                    # in this case BFF denominator == 1 and numerator is only odds. 
                    return probabilities[:, 1] / probabilities[:, 0]
                else:
                    numerator = np.prod((probabilities[:, 1] / probabilities[:, 0]).reshape(-1, self.data_sample_size), axis=1)
                    denominator = np.array([integrate.nquad(
                        func=lambda *poi: odds(self.estimator.predict_proba(
                            X=np.hstack((np.repeat(np.array([*poi]).reshape(-1, self.poi_dim), repeats=self.data_sample_size, axis=0), samples[i, :, :]))
                        ).astype(np.double)),
                        ranges=param_space_bounds[:self.poi_dim]
                    )[0] for i in tqdm(range(samples.shape[0]))])
                    assert numerator.shape == denominator.shape, f"{numerator.shape}, {denominator.shape}"
                    return numerator / denominator
            else:
                return np.array([
                    integrate.nquad(
                        func=lambda *nuisance: odds(self.estimator.predict_proba(
                            X=np.hstack((np.repeat(np.array([*parameters[i, :self.poi_dim], *nuisance]).reshape(-1, self.param_dim), repeats=self.data_sample_size, axis=0), samples[i, :, :]))
                        ).astype(np.double)),
                        ranges=param_space_bounds[-self.nuisance_dim:] 
                    )[0] / 
                    integrate.nquad(
                        func=lambda *params: odds(self.estimator.predict_proba(
                            X=np.hstack((np.repeat(np.array([*params]).reshape(-1, self.param_dim), repeats=self.data_sample_size, axis=0), samples[i, :, :]))
                        ).astype(np.double)),
                        ranges=param_space_bounds
                    )[0]
                    for i in tqdm(range(samples.shape[0]))    
                ])
        else:
            raise NotImplementedError
    
    def _compute_for_confidence_sets(
        self, 
        parameter_grid: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        params_samples = preprocess_odds_cs(parameter_grid, samples, self.param_dim, self.data_sample_size, self.estimator.model)
        if (self.null_hp_test_type == 'simple') and (self.data_sample_size == 1):
            # in this case BFF denominator == 1 and numerator is only odds. 
            # N.B.: Technically we also need G to be the marginal of F_\theta
            probabilities = self.estimator.predict_proba(X=params_samples)
            return (probabilities[:, 1] / probabilities[:, 0]).reshape(-1, parameter_grid.reshape(-1, self.param_dim).shape[0])
        else:
            raise NotImplementedError

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

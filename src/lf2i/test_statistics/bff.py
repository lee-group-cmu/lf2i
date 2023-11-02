from typing import Union, Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
from scipy import integrate
import torch
from sbi.simulators.simutils import tqdm_joblib

from lf2i.test_statistics._base import TestStatistic
from lf2i.utils.odds_inputs import (
    preprocess_odds_estimation, 
    preprocess_for_odds_cv, 
    preprocess_for_odds_cs, 
    preprocess_odds_integration
)
from lf2i.utils.miscellanea import to_np_if_torch


class BFF(TestStatistic):
    """Implements the `BFF` test statistic as described in https://arxiv.org/abs/2107.03920.
    NOTE: for now supports only box uniform proposal distributions over the parameter space. 

    Parameters
    ----------
    estimator : Union[str, Any]
        Probabilistic classifier used to estimate odds (i.e., likelihood up to a normalization constant). 
        If `str`, must be one of the predefined estimators listed in `test_statistics/_estimators.py`.
        If `Any`, a trained estimator is expected. Needs to implement `estimator.predict_proba(X=...)`.
    poi_dim : int
        Dimensionality (number) of the parameters of interest.
    nuisance_dim : int
        Dimensionality (number) of the nuisance parameters (systematics). Should be 0 if all parameters are object of inference.
    batch_size : int
        Size of a batch of datapoints from a specific parameter configuration. Must be the same for observations and simulations.
        A simulated/observed batch from a specific parameter configuration will have dimensions `(batch_size, data_dim)`.
    data_dim : int
        Dimensionality of a single datapoint X.
    estimator_kwargs : Dict, optional
        Hyperparameters and settings for the conditional mean estimator, by default {}.
    n_jobs : int, optional
        Number of workers to use when computing BFF over multiple inputs, by default -2, which uses all cores minus one.
    """

    def __init__(
        self,
        estimator: Union[str, Any],
        poi_dim: int,
        nuisance_dim: int,
        batch_size: int,
        data_dim: int,
        estimator_kwargs: Dict = {},
        n_jobs: int = -2
    ) -> None:
        super().__init__(acceptance_region='right', estimation_method='likelihood')

        self.poi_dim = poi_dim
        self.nuisance_dim = nuisance_dim
        self.param_dim = poi_dim + nuisance_dim
        self.batch_size = batch_size
        self.data_dim = data_dim
        self.estimator = self._choose_estimator(estimator, estimator_kwargs, 'odds')
        self.n_jobs = n_jobs

    def estimate(
        self,
        labels: Union[np.ndarray, torch.Tensor], 
        parameters: Union[np.ndarray, torch.Tensor], 
        samples: Union[np.ndarray, torch.Tensor],
    ) -> None:
        r"""Train the estimator for odds (i.e. likelihood up to a normalization constant).
        The training dataset should contain two classes:
            - label 1, with pairs :math:`(\theta, X)` where :math:`X \sim p(\cdot;\theta)` is drawn from the likelihood/simulator.
            - label 0, with pairs :math:`(\theta, X)` where :math:`X \sim G` is drawn from a dominating reference distribution (e.g., empirical marginal).
        To goal is to train a classifier that is able to distinguish whether a sample comes from the likelihood or not.
        See https://arxiv.org/abs/2107.03920 for a more detailed explanation.

        Parameters
        ----------
        labels : Union[np.ndarray, torch.Tensor]
            Class labels 0/1.
        parameters : Union[np.ndarray, torch.Tensor]
            Simulated parameters to be used for training.
        samples : Union[np.ndarray, torch.Tensor]
            Simulated samples to be used for training.
        """
        labels, params_samples = preprocess_odds_estimation(labels, parameters, samples, self.param_dim, self.estimator)
        self.estimator.fit(X=params_samples, y=labels)
        self._estimator_trained['odds'] = True

    def evaluate(
        self,
        parameters: Union[np.ndarray, torch.Tensor],
        samples:  Union[np.ndarray, torch.Tensor],
        mode: str,
        param_space_bounds: Optional[List[Tuple[float]]] = None
    ) -> np.ndarray:
        r"""Evaluate the BFF test statistic over the given parameters and samples. 
        Behaviour differs depending on mode: 
            - 'critical_values' and 'diagnostics' compute BFF once for each pair :math:`(\theta, X)`.
            - 'confidence_sets' computes BFF over all pairs given by the cartesian product of `parameters` (the parameter grid to construct confidence sets) and `samples`. 

        Parameters
        ----------
        parameters : Union[np.ndarray, torch.Tensor]
            Parameters over which to evaluate the test statistic.
        samples : Union[np.ndarray, torch.Tensor]
            Samples over which to evaluate the test statistic.
        mode : str
            Either 'critical_values', 'confidence_sets', 'diagnostics'.
        param_space_bounds : Optional[List[Tuple[float]]]
            Bounds of the parameter space, both POIs and nuisances. Must be in the same order as in `parameters`.
            NOTE: Bounds are needed because we support only box uniform proposal distributions over the parameter space at the moment.

        Returns
        -------
        np.ndarray
            BFF test statistics evaluated over parameters and samples.

        Raises
        ------
        ValueError
            If `mode` is not among the pre-specified values.
        """
        if mode == 'critical_values':
            return self._compute_for_critical_values(parameters, samples, param_space_bounds)
        elif mode == 'confidence_sets':
            return self._compute_for_confidence_sets(parameters, samples, param_space_bounds)
        elif mode == 'diagnostics':
            return self._compute_for_diagnostics(parameters, samples, param_space_bounds)
        else:
            raise ValueError(f"Only `critical_values`, `confidence_sets`, and `diagnostics` are supported, got {mode}")

    def _odds(
        self,
        probs: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        probs = to_np_if_torch(probs)
        return np.prod((probs[:, 1] / probs[:, 0]).reshape(-1, self.batch_size), axis=1)

    def _integrate_odds(
        self,
        sample: Union[np.ndarray, torch.Tensor],
        fixed_poi: Union[np.ndarray, torch.Tensor],  # needed only if integrating solely over nuisances; otherwise empty array
        integration_bounds: List[Tuple[float]]
    ) -> float:
        return integrate.nquad(
            func=lambda *params: self._odds(self.estimator.predict_proba(
                X=preprocess_odds_integration(self.estimator, fixed_poi, params, sample, self.param_dim, self.batch_size)
            )),
            ranges=integration_bounds
        )[0]  # return only the result of the integration
        
    def _compute_for_critical_values(
        self,
        parameters: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor],
        param_space_bounds: Optional[List[Tuple[float]]] = None
    ) -> np.ndarray:
        # TODO: 
        # 1) fix usage of param_space_bounds, allow integration over any prior;
        # 2) check if better to do computations with log (see last appendix in lf2i paper)
        # 3) optimize (minimize calls to predict_proba, vectorize, numba, parallelize); 
        # 4) check and fix consistency of types for arrays-estimators
        # NOTE: this only considers simple null hypothesis with respect to the POI, which is what we need for confidence sets
        parameters, samples, params_samples = preprocess_for_odds_cv(parameters, samples, self.param_dim, self.batch_size, self.data_dim, self.estimator)
        if self.nuisance_dim == 0:
            if self.batch_size == 1:
                # TODO: technically we also need G to be the marginal of F_\theta, and proportion of Y=1 to be 0.5
                # in this case BFF denominator == 1 and numerator is only odds. 
                return self._odds(self.estimator.predict_proba(X=params_samples))
            else:
                numerator = self._odds(self.estimator.predict_proba(X=params_samples))
                with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Computing BFF for {len(it)} points...", total=len(it))) as _:
                    denominator = np.array(Parallel(n_jobs=self.n_jobs)(delayed(
                        lambda idx: self._integrate_odds(sample=samples[idx, :, :], fixed_poi=torch.empty(0), integration_bounds=param_space_bounds[:self.poi_dim]) 
                        )(i) for i in it
                    ))
                return numerator / denominator
        else:
            def do_one(idx: int) -> float:
                num = self._integrate_odds(sample=samples[idx, :, :], fixed_poi=parameters[idx, :self.poi_dim], integration_bounds=param_space_bounds[-self.nuisance_dim:])
                den = self._integrate_odds(sample=samples[idx, :, :], fixed_poi=torch.empty(0), integration_bounds=param_space_bounds)
                return num / den
            
            with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Computing BFF for {len(it)} points...", total=len(it))) as _:
                bff = np.array(Parallel(n_jobs=self.n_jobs)(delayed(do_one)(i) for i in it))
            return bff
    
    def _compute_for_confidence_sets(
        self, 
        parameter_grid: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor],
        param_space_bounds: List[List[float]]
    ) -> np.ndarray:
        # TODO: can we call _compute_for_critical_values instead of rewriting much similar code?
        # TODO: double-check this
        parameter_grid, samples, param_grid_samples = preprocess_for_odds_cs(parameter_grid, samples, self.poi_dim, self.batch_size, self.data_dim, self.estimator)
        if self.nuisance_dim == 0:
            if self.batch_size == 1:
                # TODO: see same point in _compute_for_critical_values
                return self._odds(self.estimator.predict_proba(X=param_grid_samples)).reshape(samples.shape[0], parameter_grid.shape[0])
            else:
                numerator = self._odds(self.estimator.predict_proba(X=param_grid_samples)).reshape(samples.shape[0], parameter_grid.shape[0])
                # denominator is the same regardless of parameter grid value
                with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Computing BFF for {len(it)} points...", total=len(it))) as _:
                    denominator = np.array(Parallel(n_jobs=self.n_jobs)(delayed(
                        lambda idx: self._integrate_odds(sample=samples[idx, :, :], fixed_poi=torch.empty(0), integration_bounds=param_space_bounds[:self.poi_dim]) 
                        )(i) for i in it
                    )).reshape(-1, 1)
                return numerator / denominator  # automatic broadcasting along dimension 1
        else:
            def param_grid_loop(sample: Union[np.ndarray, torch.Tensor], denominator: float) -> np.ndarray:
                numerator = np.empty(shape=(parameter_grid.shape[0], ))
                for j in range(parameter_grid.shape[0]):
                    numerator[j] = self._integrate_odds(sample=sample, fixed_poi=parameter_grid[j, :], integration_bounds=param_space_bounds[-self.nuisance_dim:])
                return numerator / denominator
            
            with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Computing BFF for {len(it)}x{parameter_grid.shape[0]} points...", total=len(it))) as _:
                out = np.vstack(Parallel(n_jobs=self.n_jobs)(delayed(lambda idx: param_grid_loop(
                    sample=samples[idx, :, :], 
                    denominator=self._integrate_odds(sample=samples[idx, :, :], fixed_poi=torch.empty(0), integration_bounds=param_space_bounds)
                    ).reshape(1, -1))(i) for i in it
                ))
            return out

    def _compute_for_diagnostics(
        self,
        parameters: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor],
        param_space_bounds: List[List[float]]
    ) -> np.ndarray:
        return self._compute_for_critical_values(parameters, samples, param_space_bounds)

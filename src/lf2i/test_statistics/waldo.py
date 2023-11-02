from typing import Optional, Union, List, Dict, Any
import warnings
from joblib import Parallel, delayed

from tqdm import tqdm
import numpy as np
import torch
from sbi.simulators.simutils import tqdm_joblib
from lf2i.test_statistics._base import TestStatistic
from lf2i.utils.waldo_inputs import preprocess_waldo_estimation, preprocess_waldo_evaluation, preprocess_waldo_computation


class Waldo(TestStatistic):
    """
    Implements the `Waldo` test statistic, as described in arXiv:2205.15680.

    Parameters
    ----------
    estimator : Union[str, Any]
        If `estimation_method == prediction`, then this is the conditional mean estimator.
        If `estimation_method == posterior`, then this is the posterior estimator. Currently compatible with posterior objects from SBI package (https://github.com/mackelab/sbi)

        If `str`, will use one of the predefined estimators. 
        If `Any`, a trained estimator is expected. Needs to implement `estimator.predict(X=...)` ("prediction"), or `estimator.sample(sample_shape=..., x=...)` ("posterior").
    poi_dim : int
        Dimensionality (number) of the parameters of interest.
    estimation_method : str
        Whether the estimator is a prediction algorithm ("prediction") or a posterior estimator ("posterior").
    num_posterior_samples : Optional[int], optional
        Number of posterior samples to draw to approximate conditional mean and variance if `estimation_method == posterior`, by default None
    cond_variance_estimator : Optional[Union[str, Any]], optional
        If `estimation_method == prediction`, then this is the conditional variance estimator, by default None
    estimator_kwargs: Dict
        Hyperparameters and settings for the conditional mean estimator, by default {}.
    cond_variance_estimator_kwargs: Dict
        Hyperparameters and settings for the conditional variance estimator, by default {}.
    n_jobs : int, optional
        Number of workers to use when evaluating Waldo over multiple inputs if using a posterior estimator. By default -2, which uses all cores minus one.
    """

    def __init__(
        self, 
        estimator: Union[str, Any],
        poi_dim: int,
        estimation_method: str,
        num_posterior_samples: Optional[int] = None,
        cond_variance_estimator: Optional[Union[str, Any]] = None,
        estimator_kwargs: Dict = {},
        cond_variance_estimator_kwargs: Dict = {},
        n_jobs: int = -2
    ) -> None:
        super().__init__(acceptance_region='left', estimation_method=estimation_method)

        self.poi_dim = poi_dim
        if estimation_method == 'prediction':
            self.estimator = self._choose_estimator(estimator, estimator_kwargs, 'conditional_mean')
            assert cond_variance_estimator is not None, "Need to specify a model to estimate the conditional variance"
            self.cond_variance_estimator = self._choose_estimator(cond_variance_estimator, cond_variance_estimator_kwargs, 'conditional_variance')
        elif estimation_method == 'posterior':
            self.estimator = self._choose_estimator(estimator, estimator_kwargs, 'posterior')
            assert num_posterior_samples is not None, "Need to specify how many samples to draw from the posterior to approximate conditional mean and variance"
            self.num_posterior_samples = num_posterior_samples
        else:
            raise ValueError(f"Waldo estimation is supported only using `prediction` algorithms or `posterior` estimators, got {estimation_method}")
        self.n_jobs = n_jobs
    
    @staticmethod
    def _compute_for_critical_values(
        parameters: np.ndarray,
        conditional_mean: Union[np.ndarray, List],
        conditional_var: Union[np.ndarray, List]
    ) -> np.ndarray:
        if parameters.shape[-1] == 1:  # parameter is 1-dimensional
            return ( (conditional_mean - parameters)**2 ) / conditional_var
        else:
            # conditional mean and var lists of arrays
            return np.array([
                ( ( conditional_mean[idx] - parameters[idx, :] ) @ np.linalg.inv(conditional_var[idx]) ) @ ( conditional_mean[idx] - parameters[idx, :] ).T
                for idx in range(len(conditional_mean))
            ]).reshape(-1, )
    
    @staticmethod
    def _compute_for_confidence_sets(
        parameter_grid: np.ndarray,
        conditional_mean: Union[np.ndarray, List],
        conditional_var: Union[np.ndarray, List]
    ) -> np.ndarray:
        # TODO: can we avoid some redundancy using np.tile only once?
        if parameter_grid.shape[-1] == 1:  # parameter is 1-dimensional
            tile_parameters = np.tile(parameter_grid, reps=(conditional_mean.shape[0], 1)).reshape(conditional_mean.shape[0], parameter_grid.shape[0])
            return ( (conditional_mean - tile_parameters)**2 ) / conditional_var
        else:
            # conditional mean and var lists of arrays
            tile_parameters = np.tile(parameter_grid, reps=(len(conditional_mean), 1, 1)).reshape(len(conditional_mean), parameter_grid.shape[0], parameter_grid.shape[-1])
            return np.vstack([
                np.sum(( (conditional_mean[idx] - tile_parameters[idx, :, :]) @ np.linalg.inv(conditional_var[idx]) ) * (conditional_mean[idx] - tile_parameters[idx, :, :]), axis=1).reshape(-1, 1)
                for idx in range(len(conditional_mean))
            ])

    @classmethod
    def _compute_for_diagnostics(
        cls,
        parameters: np.ndarray,
        conditional_mean: Union[np.ndarray, List],
        conditional_var: Union[np.ndarray, List]
    ) -> np.ndarray:
        # same implementation
        return cls._compute_for_critical_values(parameters, conditional_mean, conditional_var)

    def _compute(
        self,
        parameters: np.ndarray,
        conditional_mean: Union[np.ndarray, List],
        conditional_var: Union[np.ndarray, List],
        mode: str
    ) -> np.ndarray:
        # TODO: unify computations regardless of self.estimation_method (prediction or posterior)
        # TODO: unify computations regardless of mode?
        # TODO: vectorize computations when d>1
        # TODO: write unit tests for all corner cases
        # TODO: if for loops are used, then we'd better switch to generators (especially for confidence sets)

        parameters, conditional_mean, conditional_var = \
            preprocess_waldo_computation(parameters, conditional_mean, conditional_var, self.poi_dim)

        if mode == 'critical_values':
            return self._compute_for_critical_values(parameters, conditional_mean, conditional_var)        
        elif mode == 'confidence_sets':
            return self._compute_for_confidence_sets(parameters, conditional_mean, conditional_var)
        elif mode == 'diagnostics':
            return self._compute_for_diagnostics(parameters, conditional_mean, conditional_var)
        else:
            raise ValueError(f"Only `critical_values`, `confidence_sets`, and `diagnostics` are supported, got {mode}")
    
    def estimate(
        self, 
        parameters: Union[np.ndarray, torch.Tensor], 
        samples: Union[np.ndarray, torch.Tensor], 
    ) -> None:
        """Train the estimator(s) for the conditional mean and conditional variance. 

        Parameters
        ----------
        parameters : Union[np.ndarray, torch.Tensor]
            Simulated parameters to be used for training.
        samples : Union[np.ndarray, torch.Tensor]
            Simulated samples to be used for training.
        """
        # if `self.estimation_method == prediction`, assume both estimators accept same input types
        parameters, samples = preprocess_waldo_estimation(parameters, samples, self.estimation_method, self.estimator, self.poi_dim)
        if self.estimation_method == 'prediction':
            self.estimator.fit(X=samples, y=parameters)
            if self.poi_dim > 1:
                warnings.warn("Using 'prediction' with poi_dim > 1 might have inconsistencies and has not been thoroughly checked yet")
            conditional_var_response = (( parameters.reshape(-1, self.poi_dim) - self.estimator.predict(X=samples).reshape(-1, self.poi_dim) )**2).reshape(-1, )
            self.cond_variance_estimator.fit(X=samples, y=conditional_var_response)
            self._estimator_trained['conditional_mean'] = True
            self._estimator_trained['conditional_variance'] = True
        else:
            _ = self.estimator.append_simulations(parameters, samples).train()
            self.estimator = self.estimator.build_posterior()
            self._estimator_trained['posterior'] = True

    def evaluate(
        self, 
        parameters: Union[np.ndarray, torch.Tensor], 
        samples: Union[np.ndarray, torch.Tensor], 
        mode: str
    ) -> np.ndarray:
        r"""Evaluate the Waldo test statistic over the given parameters and samples. 
        
        Behaviour differs depending on mode: 'critical_values', 'confidence_sets', 'diagnostics':
            - If mode equals `critical_values` or `diagnostics`, evaluate Waldo over pairs :math:`(\theta_i, x_i)`.
            - If mode equals `confidence_sets`, evaluate Waldo over all pairs given by the cartesian product of `parameters` (the parameter grid to construct confidence sets) and `samples`.

        Parameters
        ----------
        parameters : np.ndarray
            Parameters over which to evaluate the test statistic.
        samples : np.ndarray
            Samples over which to evaluate the test statistic.
        mode : str
            Either 'critical_values', 'confidence_sets', 'diagnostics'.

        Returns
        -------
        np.ndarray
            Waldo test statistics evaluated over parameters and samples.
        """
        assert self._check_is_trained(), "Not all needed estimators are trained. Check self._estimator_trained"
        # if `self.estimation_method == prediction`, assume both estimators accept same input types
        parameters, samples = preprocess_waldo_evaluation(parameters, samples, self.estimation_method, self.estimator, self.poi_dim)

        if self.estimation_method == 'prediction':
            conditional_mean = self.estimator.predict(X=samples)
            conditional_var = self.cond_variance_estimator.predict(X=samples)
        else:
            def sampling_loop(idx):
                posterior_samples = self.estimator.sample(sample_shape=(self.num_posterior_samples, ), x=samples[idx, ...], show_progress_bars=False).numpy()
                cond_mean = np.mean(posterior_samples, axis=0).reshape(1, self.poi_dim)
                cond_var = np.cov(posterior_samples.T)  # need samples.shape = (data_d, num_samples)
                return cond_mean, cond_var
            with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Approximating conditional mean and covariance for {samples.shape[0]} points...", total=len(it))) as _:
                out = list(zip(*Parallel(n_jobs=self.n_jobs)(delayed(sampling_loop)(idx) for idx in it)))  # axis 0 indexes different simulations/observations
                conditional_mean, conditional_var = out[0], out[1]
        return self._compute(parameters, conditional_mean, conditional_var, mode)


"""
def sampling_loop(idx):
    posterior_samples = self.estimator.sample(sample_shape=(self.num_posterior_samples, ), x=samples[idx, ...], show_progress_bars=False).numpy()
    cond_mean = np.mean(posterior_samples, axis=0).reshape(1, self.poi_dim)
    cond_var = np.cov(posterior_samples.T)  # need samples.shape = (data_d, num_samples)
    return cond_mean, cond_var
with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Approximating conditional mean and covariance for {samples.shape[0]} points...", total=len(it))) as _:
    out = list(zip(*Parallel(n_jobs=self.n_jobs)(delayed(sampling_loop)(idx) for idx in it)))  # axis 0 indexes different simulations/observations
conditional_mean, conditional_var = out[0], out[1]
"""

"""
conditional_mean = []
conditional_var = []
for idx in tqdm(range(samples.shape[0]), desc='Approximating conditional mean and covariance'):  # axis 0 indexes different simulations/observations
    posterior_samples = self.estimator.sample(sample_shape=(self.num_posterior_samples, ), x=samples[idx, ...], show_progress_bars=False).numpy()
    conditional_mean.append(np.mean(posterior_samples, axis=0).reshape(1, self.poi_dim))
    conditional_var.append(np.cov(posterior_samples.T))  # need samples.shape = (data_d, num_samples)
"""
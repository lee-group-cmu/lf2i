from typing import Optional, Union, List, Dict, Any
import warnings

from tqdm import tqdm
import numpy as np
import torch
from lf2i.test_statistics._base import TestStatistic
from lf2i.utils.waldo_inputs import preprocess_waldo_estimation, preprocess_waldo_evaluation, preprocess_waldo_computation


class Waldo(TestStatistic):
    """
    Implements the `Waldo` test statistic, as described in arXiv:2205.15680.

    Parameters
    ----------
    estimator : Union[str, Any]
        If `method == prediction`, then this is the conditional mean estimator.
        If `method == posterior`, then this is the posterior estimator. Currently compatible with posterior objects from SBI package (https://github.com/mackelab/sbi)

        If `str`, will use one of the predefined estimators. 
        If `Any`, a trained estimator is expected. Needs to implement `estimator.predict(X=...)` ("prediction"), or `estimator.sample(sample_shape=..., x=...)` ("posterior").
    param_dim : int
        Dimensionality of the parameters of interest
    method : str
        Whether the estimator is a prediction algorithm ("prediction") or a posterior estimator ("posterior").
    num_posterior_samples : Optional[int], optional
        Number of posterior samples to draw to approximate conditional mean and variance if `method == posterior`, by default None
    cond_variance_estimator : Optional[Union[str, Any]], optional
        If `method == prediction`, then this is the conditional variance estimator, by default None
    estimator_kwargs: Dict
        Hyperparameters and settings for the conditional mean estimator, by default {}.
    cond_variance_estimator_kwargs: Dict
        Hyperparameters and settings for the conditional variance estimator, by default {}.
    """

    def __init__(
        self, 
        estimator: Union[str, Any],
        param_dim: int,
        method: str,
        num_posterior_samples: Optional[int] = None,
        cond_variance_estimator: Optional[Union[str, Any]] = None,
        estimator_kwargs: Dict = {},
        cond_variance_estimator_kwargs: Dict = {}
    ) -> None:
        super().__init__(acceptance_region='left')

        self.method = method
        self.param_dim = param_dim
        if method == 'prediction':
            self.estimator = self._choose_estimator(estimator, estimator_kwargs, 'conditional_mean')
            assert cond_variance_estimator is not None, "Need to specify a model to estimate the conditional variance"
            self.cond_variance_estimator = self._choose_estimator(cond_variance_estimator, cond_variance_estimator_kwargs, 'conditional_variance')
        elif method == 'posterior':
            self.estimator = self._choose_estimator(estimator, estimator_kwargs, 'posterior')
            assert num_posterior_samples is not None, "Need to specify how many samples to draw from the posterior to approximate conditional mean and variance"
            self.num_posterior_samples = num_posterior_samples
        else:
            raise ValueError(f"Waldo estimation is supported only using `prediction` algorithms or `posterior` estimators, got {method}")
    
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
        """
        Compute the Waldo test statistic in a suitable way given `mode`.
        If `mode == critical_values` or `mode == diagnostics`, evaluate Waldo over pairs `\{(\theta_i, x_i)}_{i=1,\dots}`
        If `mode == confidence_sets`, evaluate Waldo over all parameters *for each* sample.

        Parameters
        ----------
        parameters : np.ndarray
            Parameters over which to evaluate the test statistic.
        conditional_mean : Union[np.ndarray, List]
            Conditioanal means (given samples), to use in the computation of Waldo.
        conditional_var : Union[np.ndarray, List]
            Conditional variances - or covariance matrices - (given samples), to use in the computation of Waldo.
        mode : str
            Either 'critical_values', 'confidence_sets', 'diagnostics'.

        Returns
        -------
        np.ndarray
            Waldo test statistics evaluated over parameters and samples.

        Raises
        ------
        ValueError
            If `mode` is not among the pre-specified values.
        """
        # TODO: unify computations regardless of self.method (prediction or posterior)
        # TODO: unify computations regardless of mode?
        # TODO: vectorize computations when d>1
        # TODO: write unit tests for all corner cases
        # TODO: if for loops are used, then we'd better switch to generators (especially for confidence sets)

        parameters, conditional_mean, conditional_var = \
            preprocess_waldo_computation(parameters, conditional_mean, conditional_var, self.param_dim)

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
        # if `self.method == prediction`, assume both estimators accept same input types
        # TODO: check that inputs have correct shapes for each method. What if data_sample_size > 1?
        parameters, samples = preprocess_waldo_estimation(parameters, samples, self.method, self.estimator, self.param_dim)
        if self.method == 'prediction':
            self.estimator.fit(X=samples, y=parameters)
            if self.param_dim > 1:
                warnings.warn("Using 'prediction' with param_dim > 1 might have inconsistencies and has not been thoroughly checked yet")
            conditional_var_response = (( parameters.reshape(-1, self.param_dim) - self.estimator.predict(X=samples).reshape(-1, self.param_dim) )**2).reshape(-1, )
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
        """Evaluate the Waldo test statistic over the given parameters and samples. 
        
        Behaviour differs depending on mode: 'critical_values', 'confidence_sets', 'diagnostics'.
        See self.compute() for details. 

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
        # if `self.method == prediction`, assume both estimators accept same input types
        parameters, samples = preprocess_waldo_evaluation(parameters, samples, self.method, self.estimator, self.param_dim)

        if self.method == 'prediction':
            conditional_mean = self.estimator.predict(X=samples)
            conditional_var = self.cond_variance_estimator.predict(X=samples)
        else:
            conditional_mean = []
            conditional_var = []
            for idx in tqdm(range(samples.shape[0]), desc='Approximating conditional mean and covariance'):  # axis 0 indexes different simulations/observations
                posterior_samples = self.estimator.sample(sample_shape=(self.num_posterior_samples, ), x=samples[idx, ...], show_progress_bars=False).numpy()
                conditional_mean.append(np.mean(posterior_samples, axis=0).reshape(1, self.param_dim))
                conditional_var.append(np.cov(posterior_samples.T))  # need samples.shape = (data_d, num_samples)
        return self._compute(parameters, conditional_mean, conditional_var, mode)

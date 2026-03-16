from typing import Union, Any, Dict, Optional, List, Tuple
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar

from lf2i.test_statistics._base import TestStatistic
from lf2i.utils.odds_inputs import (
    preprocess_odds_estimation, 
    preprocess_for_odds_cv, 
    preprocess_odds_maximization, 
    preprocess_for_odds_cs
)
from lf2i.utils.miscellanea import to_np_if_torch
from lf2i.utils.parallel import tqdm_joblib


class ACORE(TestStatistic):
    """Implements the `ACORE` test statistic as described in https://proceedings.mlr.press/v119/dalmasso20a.html and https://arxiv.org/abs/2107.03920.

    Parameters
    ----------
    estimator : Union[str, Any]
        Probabilistic classifier used to estimate odds (i.e., likelihood up to a normalization constant). 
        If `str`, must be one of the predefined estimators listed in `test_statistics/estimators.py`.
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
    verbose: bool, optional
        Whether to print progress bars when evaluating or not, by default True.
    n_jobs : int, optional
        Number of workers to use when computing ACORE over multiple inputs, by default -2, which uses all cores minus one.
        `n_jobs == -1` uses all cores. If `n_jobs < -1`, then `n_jobs = os.cpu_count()+1+n_jobs`.
    """
    
    def __init__(
        self,
        estimator: Union[str, Any],
        poi_dim: int,
        nuisance_dim: int,
        batch_size: int,
        data_dim: int,
        estimator_kwargs: Dict = {},
        verbose: bool = True,
        n_jobs: int = -2,
        param_space_bounds: List[Tuple[float]] = None,
        max_iter: Optional[int] = 1,
        estimator_train_kwargs: Optional[Dict] = None
    ) -> None:
        super().__init__(acceptance_region='right', estimation_method='likelihood')

        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        self.poi_dim = poi_dim
        self.nuisance_dim = nuisance_dim
        self.param_dim = poi_dim + nuisance_dim
        self.batch_size = batch_size
        self.data_dim = data_dim
        self.estimator = self._choose_estimator(estimator, estimator_kwargs, 'odds')
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.param_space_bounds = param_space_bounds
        self.max_iter = max_iter
        self.estimator_train_kwargs = estimator_train_kwargs

    def estimate(
        self,
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
        labels, params_samples = preprocess_odds_estimation(
            parameters, samples, self.param_dim, self.estimator
        )
        train_validate_split = int(0.9 * len(labels))
        X, y = params_samples[:train_validate_split], labels[:train_validate_split]
        X_val, y_val = params_samples[train_validate_split:], labels[train_validate_split:]
        history = self.estimator.fit(X=X, y=y, X_val=X_val, y_val=y_val, **(self.estimator_train_kwargs if self.estimator_train_kwargs is not None else {}))
        self._estimator_trained['odds'] = True
        return history

    def evaluate(
        self,
        parameters: Union[np.ndarray, torch.Tensor],
        samples:  Union[np.ndarray, torch.Tensor],
        mode: str,
        param_space_bounds: Optional[List[Tuple[float]]] = None,
        condition_on_poi: bool = False # TODO: implement evaluation mode for POI-conditional sets
    ) -> np.ndarray:
        r"""Evaluate the ACORE test statistic over the given parameters and samples. 
        Behaviour differs depending on mode: 
            - 'critical_values' and 'diagnostics' compute ACORE once for each pair :math:`(\theta, X)`.
            - 'confidence_sets' computes ACORE over all pairs given by the cartesian product of `parameters` (the parameter grid to construct confidence sets) and `samples`. 

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
        condition_on_poi : bool
            Whether to condition on the POI when maximizing the likelihood for the denominator of the ACORE

        Returns
        -------
        np.ndarray
            ACORE test statistics evaluated over parameters and samples.

        Raises
        ------
        ValueError
            If `mode` is not among the pre-specified values.
        """
        if param_space_bounds is None:
            param_space_bounds = self.param_space_bounds

        if mode == 'critical_values':
            return self._compute_for_critical_values(parameters, samples, param_space_bounds, condition_on_poi)
        elif mode == 'confidence_sets':
            return self._compute_for_confidence_sets(parameters, samples, param_space_bounds, condition_on_poi)
        elif mode == 'diagnostics':
            return self._compute_for_diagnostics(parameters, samples, param_space_bounds, condition_on_poi)
        else:
            raise ValueError(f"Only `critical_values`, `confidence_sets`, and `diagnostics` are supported, got {mode}")

    def _compute_for_critical_values(
        self,
        parameters: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor, None],
        param_space_bounds: Optional[List[Tuple[float]]],
        condition_on_poi: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        # NOTE: this only considers simple null hypothesis with respect to the POI, which is what we need for confidence sets
        parameters, samples, params_samples = preprocess_for_odds_cv(parameters, samples, self.param_dim, self.batch_size, self.data_dim, self.estimator)
        if not condition_on_poi:
            if self.nuisance_dim == 0:
                numerator = self._log_odds(self.estimator.predict_proba(X=params_samples))[:, 1]
                with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Evaluating ACORE for {len(it)} points...", total=len(it), disable=not self.verbose)) as _:
                    denominator = np.array(Parallel(n_jobs=self.n_jobs)(delayed(
                        lambda idx: self._maximize_log_odds(sample=samples[idx], fixed_poi=torch.empty(0), optimization_bounds=param_space_bounds) 
                        )(i) for i in it
                    ))
                return (numerator - denominator)
            else:
                def do_one(idx: int) -> float:
                    num = self._maximize_log_odds(sample=samples[idx], fixed_poi=parameters[idx, :self.poi_dim], optimization_bounds=param_space_bounds)
                    den = self._maximize_log_odds(sample=samples[idx], fixed_poi=torch.empty(0), optimization_bounds=param_space_bounds)
                    return (num - den)

                with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Evaluating ACORE for {len(it)} points...", total=len(it), disable=not self.verbose)) as _:
                    acore = np.array(Parallel(n_jobs=self.n_jobs)(delayed(do_one)(i) for i in it))
                return acore

        else:
            assert self.nuisance_dim > 0, "Conditioning on the POI when maximizing the likelihood for the denominator of the ACORE only makes sense if there are nuisance parameters to optimize over. Got nuisance_dim = 0."
            numerator = self._log_odds(self.estimator.predict_proba(X=params_samples))[:, 1]
            with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Evaluating ACORE for {len(it)} points...", total=len(it), disable=not self.verbose)) as _:
                denominator = np.array(Parallel(n_jobs=self.n_jobs)(delayed(
                    lambda idx: self._maximize_log_odds(sample=samples[idx], fixed_poi=parameters[idx, :self.poi_dim], optimization_bounds=param_space_bounds) 
                    )(i) for i in it
                ))
            return (numerator - denominator)

    def _compute_for_confidence_sets(
        self, 
        parameter_grid: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor],
        param_space_bounds: List[List[float]],
        condition_on_poi: bool = False
    ) -> np.ndarray:
        parameter_grid, samples, param_grid_samples = preprocess_for_odds_cs(parameter_grid, samples, self.param_dim, self.batch_size, self.data_dim, self.estimator)
        poi_grid = parameter_grid[:, :self.poi_dim]

        if not condition_on_poi:
            if self.nuisance_dim == 0:
                # log_odds already aggregates wrt batch_size
                numerator = self._log_odds(self.estimator.predict_proba(X=param_grid_samples)).reshape(samples.shape[0], parameter_grid.shape[0])
                # denominator is the same regardless of parameter grid value
                with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Computing ACORE for {len(it)} points...", total=len(it), disable=not self.verbose)) as _:
                    denominator = np.array(Parallel(n_jobs=self.n_jobs)(delayed(
                        lambda idx: self._maximize_log_odds(sample=samples[idx], fixed_poi=torch.empty(0), optimization_bounds=param_space_bounds) 
                        )(i) for i in it
                    )).reshape(-1, 1)
                return (numerator - denominator)  # automatic broadcasting along dimension 1
            else:
                def param_grid_loop(sample: Union[np.ndarray, torch.Tensor], denominator: float) -> np.ndarray:
                    numerator = np.empty(shape=(parameter_grid.shape[0], ))
                    for j in range(parameter_grid.shape[0]):
                        numerator[j] = self._maximize_log_odds(sample=sample, fixed_poi=poi_grid[j, :], optimization_bounds=param_space_bounds)
                    return (numerator - denominator)
                
                with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Computing ACORE for {len(it)}x{parameter_grid.shape[0]} points...", total=len(it), disable=not self.verbose)) as _:
                    out = np.vstack(Parallel(n_jobs=self.n_jobs)(delayed(lambda idx: param_grid_loop(
                        sample=samples[idx], 
                        denominator=self._maximize_log_odds(sample=samples[idx], fixed_poi=torch.empty(0), optimization_bounds=param_space_bounds)
                        ).reshape(1, -1))(i) for i in it
                    ))
                return out

        else:
            assert self.nuisance_dim > 0, "Conditioning on the POI when maximizing the likelihood for the denominator of the ACORE only makes sense if there are nuisance parameters to optimize over. Got nuisance_dim = 0."
            def param_grid_loop(sample: Union[np.ndarray, torch.Tensor], denominator: float) -> np.ndarray:
                numerator = np.empty(shape=(parameter_grid.shape[0], ))
                for j in range(parameter_grid.shape[0]):
                    numerator[j] = self._maximize_log_odds(sample=sample, fixed_poi=poi_grid[j, :], optimization_bounds=param_space_bounds)
                return (numerator - denominator)

            with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Computing ACORE for {len(it)}x{parameter_grid.shape[0]} points...", total=len(it), disable=not self.verbose)) as _:
                out = np.vstack(Parallel(n_jobs=self.n_jobs)(delayed(lambda idx: param_grid_loop(
                    sample=samples[idx], 
                    denominator=self._maximize_log_odds(sample=samples[idx], fixed_poi=poi_grid[idx, :], optimization_bounds=param_space_bounds)
                    ).reshape(1, -1))(i) for i in it
                ))
            return out

    def _compute_for_diagnostics(
        self,
        parameters: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor],
        param_space_bounds: List[List[float]],
        condition_on_poi: bool = False
    ) -> np.ndarray:
        return self._compute_for_critical_values(parameters, samples, param_space_bounds)

    def _log_odds(
        self,
        prob: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Convert odds-scale input to log-likelihood-scale
        """
        prob = np.clip(to_np_if_torch(prob), 1e-6, 1 - 1e-6)  # avoid numerical issues with log(0) or log(inf)
        return np.log(prob / (1 - prob))

    def _log_lik(
        self,
        parameter: Union[np.ndarray, torch.Tensor],
        sample: Union[np.ndarray, torch.Tensor]
    ):
        """
        Evaluate the log-likelihood (up to a normalization constant) for a given parameter and sample, using the trained estimator for odds.
        """
        return self._log_odds(self.estimator.predict_proba(
            X=preprocess_odds_maximization(self.estimator, parameter, parameter[0], 0, sample)
        ))[0, 1].item()

    def _maximize_log_odds(
        self,
        sample: Union[np.ndarray, torch.Tensor],
        fixed_poi: Union[np.ndarray, torch.Tensor],  # needed only if maximizing solely over nuisances; otherwise empty array
        optimization_bounds: List[Tuple[float]],
        argmax: bool = False,
        # max_iter: Optional[int] = 1,
        condition_on_poi: bool = False # TODO: implement conditioning on the POI when maximizing the likelihood for the denominator of the ACORE
    ) -> float:
        assert fixed_poi.shape[0] in [0, self.poi_dim], f"fixed_poi should be either empty or have the same number of dimensions as the number of POIs, got {fixed_poi.shape[0]} and {self.poi_dim} respectively"

        # Set nominal parameter based on global or restricted MLE
        if fixed_poi.shape[0] > 0:
            opt_dims = range(self.poi_dim, self.param_dim)
            nominal_parameter = torch.cat((fixed_poi, torch.tensor(
                np.array([np.mean(bounds) for bounds in optimization_bounds[self.poi_dim:]])
            )))  # use mid-point as initial guess for nuisances
        else:
            opt_dims = range(self.param_dim)
            nominal_parameter = torch.tensor(
                np.array([np.mean(bounds) for bounds in optimization_bounds])
            )  # use mid-point as initial guess

        for iteration in range(self.max_iter):
            current_nominal_parameter = nominal_parameter.clone()  # keep track of the current nominal parameter to check for convergence

            for pdx in opt_dims:
                # Profile of likelihood along parameter dimension pdx
                def objective(theta_j: float) -> float:
                    return -1 * self._log_odds(self.estimator.predict_proba(
                        X=preprocess_odds_maximization(self.estimator, current_nominal_parameter, theta_j, pdx, sample)
                    ))[0, 1].item()

                result = minimize_scalar(
                    objective,
                    bounds=optimization_bounds[pdx],
                    method='bounded'
                )
                current_nominal_parameter[pdx] = result.x

            nominal_parameter = current_nominal_parameter.clone()

        if argmax:
            return nominal_parameter
        else:
            return self._log_lik(nominal_parameter, sample)

    def _marginalize_log_odds(
        self,
        sample: Union[np.ndarray, torch.Tensor],
        fixed_poi: Union[np.ndarray, torch.Tensor],
        optimization_bounds: List[Tuple[float]],
        # max_iter: Optional[int] = 1,
        condition_on_poi: bool = False # TODO: implement conditioning on the POI when maximizing the likelihood for the denominator of the ACORE
    ) -> float:
        """
        Algorithm
        - Compute the max a posteriori estimator of via log p(x; mu, nu) + log f(mu)
        - Approximate the determinant of the Hessian of the negative log-posterior at the MAP estimator on diagonal terms via finite differences
        - Use Laplace approximation to compute the marginal likelihood,
            log p(x; mu) = log p(x; mu, nu_hat) + log f(nu_hat) + (1/2) * log(2 * pi) - (1/2) * log(det(H_(mu, mu)(nu_hat))))
         where nu_hat is the MAP estimator of the nuisance parameters, d is the number of nuisance parameters, and H is the Hessian of the negative log-posterior at the MAP estimator.
         See https://en.wikipedia.org/wiki/Laplace%27s_method_(statistics) for more details on Laplace approximation.
        """
        # TODO
        raise NotImplementedError("Marginalization of the likelihood over the nuisance parameters is not yet implemented. This would require integrating the likelihood over the nuisance parameters, which can be done via Monte Carlo integration or other numerical methods. For now, only profiling (maximization) is implemented for handling nuisance parameters in ACORE.")

    def _compute_restricted_mle_for_confidence_sets(
        self,
        parameter_grid: Union[np.ndarray, torch.Tensor],
        samples: Union[np.ndarray, torch.Tensor],
        param_space_bounds: List[List[float]]
    ) -> np.ndarray:
        parameter_grid, samples, param_grid_samples = preprocess_for_odds_cs(parameter_grid, samples, self.param_dim, self.batch_size, self.data_dim, self.estimator)
        poi_grid = parameter_grid[:, :self.poi_dim]

        if self.nuisance_dim == 0:
            # log_odds already aggregates wrt batch_size
            mle = self._log_odds(self.estimator.predict_proba(X=param_grid_samples), argmax=True).reshape(samples.shape[0], parameter_grid.shape[0])
            return mle
        else:
            def param_grid_loop(sample: Union[np.ndarray, torch.Tensor], denominator: float) -> np.ndarray:
                mle = np.empty(shape=(parameter_grid.shape[0], self.param_dim))
                for j in range(parameter_grid.shape[0]):
                    mle[j, :] = self._maximize_log_odds(sample=sample, fixed_poi=poi_grid[j, :], optimization_bounds=param_space_bounds, argmax=True)
                return mle
            
            with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Computing ACORE for {len(it)}x{parameter_grid.shape[0]} points...", total=len(it), disable=not self.verbose)) as _:
                out = np.vstack(Parallel(n_jobs=self.n_jobs)(delayed(lambda idx: param_grid_loop(
                    sample=samples[idx], 
                    denominator=self._maximize_log_odds(sample=samples[idx], fixed_poi=torch.empty(0), optimization_bounds=param_space_bounds)
                    ).reshape(1, parameter_grid.shape[0], self.param_dim))(i) for i in it
                ))
            return out

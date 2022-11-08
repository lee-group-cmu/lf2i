from typing import Optional, Union, Dict, List, Tuple, Any

import numpy as np
import torch

from lf2i.simulator._base import Simulator
from lf2i.test_statistics._base import TestStatistic
from lf2i.test_statistics.waldo import Waldo
from lf2i.critical_values.quantile_regression import train_qr_algorithm
from lf2i.confidence_regions.neyman_inversion import compute_confidence_regions
from lf2i.diagnostics.diagnostics import (
    coverage_diagnostics, 
    compute_indicators_lf2i, 
    compute_indicators_posterior, 
    compute_indicators_prediction
)


class LF2I:
    """Entry point to do inference with LF2I. 

    Once a simulator has been defined, a test statistic needs to be chosen according to the underlying inferential model that is being used.
    For example, `ACORE` and `BFF` might be chosen if one wants to estimate the likelihood. 
    Conversely, `WALDO` should be chosen if the underlying model is a prediction algorithm or a posterior estimator. See `WALDO` below for a dedicated entry point.

    Parameters
    ----------
    simulator : Simulator
        Instance of simulator object which adheres to the specifications of `lf2i.simulator._base.Simulator`
    test_statistic : TestStatistic
        Instance of test statistic object which adheres to the specifications of `lf2i.test_statistics._base.TestStatistic`
    confidence_level : float
        Desired confidence level
    """

    def __init__(
        self,
        simulator: Simulator,
        test_statistic: TestStatistic,
        confidence_level: float
    ) -> None:
        
        self.simulator = simulator
        self.test_statistic = test_statistic
        self.quantile_regressor = None
        self.critical_values = None
        self.diagnostics_estimator = None
        self.confidence_level = confidence_level

    def infer(
        self,
        x: Union[np.ndarray, torch.Tensor],
        b: int, 
        b_prime: int,
        quantile_regressor: Union[str, Any] = 'gb',
        quantile_regressor_kwargs: Dict = {},
        re_estimate_test_statistics: bool = False,
        re_estimate_critical_values: bool = False
    ) -> List[np.ndarray]:
        """Estimate the test statistics and the critical values, and construct a confidence region for all observations in `x`.

        Parameters
        ----------
        x : Union[np.ndarray, torch.Tensor]
            Observed sample(s).
        b : int
            Number of simulations to estimate the test statistics.
        b_prime : int
            Number of simulations to estimate the critical values.
        quantile_regressor : Union[str, Any], optional
            If `str`, it is an identifier for the quantile regressor to use, by default 'gb'.
            If `Any`, must be a quantile regressor with `.fit(X=..., y=...)` and `.predict(X=...)` methods.
            Currently available: ['gb', 'nn']
        quantile_regressor_kwargs : Dict, optional
            Settings for the chosen quantile regressor, by default {}
        re_estimate_test_statistics : bool, optional
            Whether to re-estimate the test statistics if a previous call to `.infer()` was made, by default False.
        re_estimate_critical_values : bool, optional
            Whether to re-estimate the critical values if a previous call to `.infer()` was made, by default False.

        Returns
        -------
        List[np.ndarray]
            The `i`-th element is a confidence region for the `i`-th sample in `x`.
        """
        if (quantile_regressor == 'gb') and (quantile_regressor_kwargs == {}):
            quantile_regressor_kwargs = {'n_estimators': 500, 'max_depth': 1}

        # estimate test statistics
        if (not self.test_statistic._check_is_trained()) or re_estimate_test_statistics:
            parameters_ts, samples_ts = self.simulator.simulate_for_test_statistic(b)
            self.test_statistic.estimate(parameters_ts, samples_ts)
            
        # estimate critical values
        if (self.quantile_regressor is None) or re_estimate_critical_values:
            parameters_cv, samples_cv = self.simulator.simulate_for_critical_values(b_prime)
            test_statistics_cv = self.test_statistic.evaluate(parameters_cv, samples_cv, mode='critical_values')
            self.quantile_regressor, self.critical_values = train_qr_algorithm(  # TODO: decouple training from prediction
                test_statistics=test_statistics_cv,
                parameters=parameters_cv,
                algorithm=quantile_regressor,
                algorithm_kwargs=quantile_regressor_kwargs,
                prediction_grid=self.simulator.param_grid,
                alpha=self.confidence_level,  # TODO: is this correct for any test statistics? For ACORE, e.g., I think it should be 1-self.confidence_level
                param_dim=self.simulator.param_dim
            )

        # construct confidence_regions
        test_statistics_x = self.test_statistic.evaluate(self.simulator.param_grid, x, mode='confidence_sets')
        confidence_regions = compute_confidence_regions(
            test_statistic=test_statistics_x,
            critical_values=self.critical_values,
            parameter_grid=self.simulator.param_grid,
            acceptance_region=self.test_statistic.acceptance_region,
            param_dim=self.simulator.param_dim
        )
        return confidence_regions

    def diagnose(
        self,
        b_doubleprime: int,
        region_type: Union[str, None],
        proba_estimator: str = 'gam',
        proba_estimator_kwargs: Dict = {},
        new_parameters: Optional[np.ndarray] = None,
        indicators: Optional[np.ndarray] = None,
        parameters: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Independent diagnostics check for the empirical conditional coverage of the desired parameter regions.

        Parameters
        ----------
        b_doubleprime : int
            Number of simulations to estimate the conditional coverage probability.
        region_type : Union[str, None]
            Whether the parameter regions to be checked are confidence regions from LF2I ('lf2i'), 
            credible regions from a posterior distribution ('posterior') or central (Gaussian approximation) prediction intervals ('prediction').
            If `posterior` or `prediction`, then `self.test_statistic` must be an instance of `Waldo`. If not, then must provide `indicators` and `parameters`.
            If `None`, then must provide `indicators` and `parameters`.
        proba_estimator : str, optional
            Identifier for the probabilistic classifier to use to estimate conditional coverage probabilities, by default 'gam'
        proba_estimator_kwargs : Dict, optional
            Settings for the probabilistic classifier, by default {}
        new_parameters : Optional[np.ndarray], optional
            If provided, coverage probabilities are estimated conditional on these parameters, by default None.
            If `None`, parameters simulated uniformly over the parameter space are used.
        indicators : Optional[np.ndarray], optional
            Pre-computed indicators (0-1) that mark whether the corresponding value in `parameters` is included or not in the target parameter region, by default None
        parameters : Optional[np.ndarray], optional
            Array of parameters for which the corresponding `indicators` have been pre-computed, by default None

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Evaluated parameters and estimated conditional coverage probabilities -- mean, upper-n_sigma bound. lower-n_sigma bound

        Raises
        ------
        ValueError
            If both `region_type is None` and `indicators is None` 
        NotImplementedError
            If `region_type not in ['lf2i', 'posterior']`
        """
        if indicators is None:
            parameters, samples = self.simulator.simulate_for_diagnostics(b_doubleprime)
        
            if region_type == 'lf2i':
                indicators = compute_indicators_lf2i(
                    test_statistics=self.test_statistic.evaluate(parameters, samples, mode='diagnostics'),
                    critical_values=self.quantile_regressor.predict(X=parameters),
                    parameters=parameters,
                    acceptance_region=self.test_statistic.acceptance_region,
                    param_dim=self.simulator.param_dim
                )
            elif region_type == 'posterior':
                assert isinstance(self.test_statistic, Waldo), \
                        "Test statistic is not an instance of `Waldo`. You must provide `indicators` and `parameters` to diagnose posteriors."
                indicators = compute_indicators_posterior(
                    posterior=self.test_statistic.estimator,
                    parameters=parameters,
                    samples=samples,
                    parameter_grid=self.simulator.param_grid,
                    confidence_level=self.confidence_level,
                    param_dim=self.simulator.param_dim,
                    data_sample_size=self.simulator.data_sample_size,
                    num_p_levels=1_000
                )
            elif region_type == 'prediction':
                assert isinstance(self.test_statistic, Waldo), \
                        "Test statistic is not an instance of `Waldo`. You must provide `indicators` and `parameters` to diagnose prediction sets."
                indicators = compute_indicators_prediction(
                    test_statistic=self.test_statistic,
                    parameters=parameters,
                    samples=samples,
                    confidence_level=self.confidence_level,
                    param_dim=self.simulator.param_dim
                )
            elif region_type is None:
                raise ValueError(
                    """If the parameter regions you want to diagnose are not from LF2I, nor they are posterior credible regions or\n 
                    central (Gaussian approximation) prediction sets, then you must provide `indicators` and `parameters`"""
                )
            else:
                raise NotImplementedError
        
        # TODO: re-use previously trained probabilistic classifier if desired
        self.diagnostics_estimator, out_parameters, mean_proba, upper_proba, lower_proba = coverage_diagnostics(
            indicators=indicators,
            parameters=parameters,
            estimator=proba_estimator,
            estimator_kwargs=proba_estimator_kwargs,
            param_dim=self.simulator.param_dim,
            new_parameters=new_parameters
        )
        return out_parameters, mean_proba, upper_proba, lower_proba


class WALDO(LF2I):
    """Entry point to do inference with WALDO. 

    Once a simulator has been defined and a prediction algorithm or posterior estimator has been chosen, this class allows for one-line inference given some observations `x`.

    Parameters
    ----------
    simulator : Simulator
        Instance of simulator object which adheres to the specifications of `lf2i.simulator._base.Simulator`
    estimator : Union[str, Any]
        Which estimator to use for the conditional mean of the test statistics. 
        
        If `str` and `method == "prediction"`, must be one of ['gb', 'rf', 'mlp']. If `method == "posterior"`, must be one of ['snpe'].
        In this case, this estimator is for the conditional mean and one needs to specify an additional one for the conditional variance.
        
        If `Any` and `method == "prediction"`, must have `.fit(X=..., y=...)` and `predict(X=...)` methods. 
        If `method == "posterior"`, we currently support posterior estimators from the SBI package (https://github.com/mackelab/sbi).
    method : str
        Whether the underlying estimator is a prediction algorithm ("prediction") or a posterior estimator ("posterior").
    confidence_level : float
        Desired confidence level.
    num_posterior_samples : Optional[int], optional
        Number of posterior samples to draw to approximate conditional mean and variance if `method == "posterior"`, by default None.
    conditional_variance_estimator : Optional[Union[str, Any]], optional
        Which estimator to use for the conditional variance (or covariance) of the test statistics, by default None.
        Only used if `method == "prediction"`. See explanation for `estimator` for details on supported estimators.
    estimator_kwargs : Dict, optional
        Settings for the chosen estimator, by default {}.
    cond_variance_estimator_kwargs : Dict, optional
        Settings for the chosen estimator for the conditional variance (or covariance), by default {}.
        Only used if `method == "prediction"`.
    """

    def __init__(
        self,
        simulator: Simulator,
        estimator: Union[str, Any],
        method: str,
        confidence_level: float,
        num_posterior_samples: Optional[int] = None,
        conditional_variance_estimator: Optional[Union[str, Any]] = None,
        estimator_kwargs: Dict = {},
        cond_variance_estimator_kwargs: Dict = {}
    ) -> None:

        self.simulator = simulator
        self.test_statistic = Waldo(
            estimator=estimator,
            param_dim=self.simulator.param_dim,
            method=method,
            num_posterior_samples=num_posterior_samples,
            cond_variance_estimator=conditional_variance_estimator,
            estimator_kwargs=estimator_kwargs,
            cond_variance_estimator_kwargs=cond_variance_estimator_kwargs
        )
        self.quantile_regressor = None
        self.diagnostics_estimator = None
        self.confidence_level = confidence_level

from typing import Optional, Union, Dict, List, Tuple, Any
import warnings

import numpy as np
import torch

from lf2i.simulator import Simulator
from lf2i.test_statistics import TestStatistic, ACORE, BFF, Waldo
from lf2i.critical_values.quantile_regression import train_qr_algorithm
from lf2i.confidence_regions.neyman_inversion import compute_confidence_regions
from lf2i.diagnostics.diagnostics import (
    estimate_coverage_proba, 
    compute_indicators_lf2i, 
    compute_indicators_posterior, 
    compute_indicators_prediction
)
from lf2i.utils.calibration_diagnostics_inputs import preprocess_predict_quantile_regression
from lf2i.utils.miscellanea import to_np_if_torch


class LF2I:
    """
    High-level entry point to do inference with LF2I (https://arxiv.org/abs/2107.03920). 
    This allows to quickly construct confidence regions for parameters of interest in an SBI setting leveraging an arbitrary estimator
        - of the *likelihood*, using for example the ACORE or BFF test statistics (https://arxiv.org/pdf/2002.10399.pdf, https://arxiv.org/abs/2107.03920);
        - of the *posterior*, using for example the Waldo test statistic (https://arxiv.org/abs/2205.15680);
        - of point estimates, i.e. a general *prediction* algorithm, using again the WALDO test statistic.
    Alternatively, one can define a custom `TestStatistic` appropriate for the problem at hand.

    NOTE: although this entry point contains all the main LF2I functionalities, using the single implemented components (test statistics, critical values, neyman inversion)
    provides a bit more flexibility and allows to control every single hyper-parameter.

    Parameters
    ----------
    test_statistic : Union[str, TestStatistic]
        Either `acore`, `bff`, `waldo` or an instance of a custom `lf2i.test_statistics._base.TestStatistic`
    test_statistic_kwargs: Any
        Arguments specific to the chosen test statistic if one of `acore`, `bff`, `waldo`. See the dedicated documentation for each of them in lf2i/test_statistics/
    """

    def __init__(
        self,
        test_statistic: Union[str, TestStatistic],
        **test_statistic_kwargs: Any
    ) -> None:
    
        if test_statistic == 'acore':
            self.test_statistic = ACORE(**test_statistic_kwargs)
        elif test_statistic == 'bff':
            self.test_statistic = BFF(**test_statistic_kwargs)
        elif test_statistic == 'waldo':
            self.test_statistic = Waldo(**test_statistic_kwargs)
        elif isinstance(test_statistic, TestStatistic):
            self.test_statistic = test_statistic
        else:
            raise ValueError(f"Expected one of `acore`, `bff`, `waldo` or an instance of a custom `lf2i.test_statistics._base.TestStatistic`, got {test_statistic}")
        self.quantile_regressor = None

    def inference(
        self,
        x: Union[np.ndarray, torch.Tensor],
        evaluation_grid: Union[np.ndarray, torch.Tensor],
        confidence_level: float,
        quantile_regressor: Union[str, Any] = 'gb',
        quantile_regressor_kwargs: Dict = {},
        T: Optional[Tuple[Union[np.ndarray, torch.Tensor]]] = None,
        T_prime: Optional[Tuple[Union[np.ndarray, torch.Tensor]]] = None,
        simulator: Optional[Simulator] = None,
        b: Optional[int] = None, 
        b_prime: Optional[int] = None, 
        re_estimate_test_statistics: bool = False,
        re_estimate_critical_values: bool = False
    ) -> List[np.ndarray]:
        """Estimate test statistic and critical values, and construct a confidence region for all observations in `x`.

        Parameters
        ----------
        x : Union[np.ndarray, torch.Tensor]
            Observed sample(s).
        evaluation_grid: Union[np.ndarray, torch.Tensor]
            Grid of points over the parameter space over which to invert hypothesis tests and construct the confidence regions. 
            Each confidence set will be a subset of this grid.
        confidence_level : float
            Desired confidence level, must be in (0, 1).
        quantile_regressor : Union[str, Any], optional
            If `str`, it is an identifier for the quantile regressor to use, by default 'gb'.
            If `Any`, must be a quantile regressor with `.fit(X=..., y=...)` and `.predict(X=...)` methods.
            Currently available: ['gb', 'nn']
        quantile_regressor_kwargs : Dict, optional
            Settings for the chosen quantile regressor, by default {}
        T: Tuple[Union[np.ndarray, torch.Tensor]], optional
            Simulated dataset to train the estimator for the test statistic. Must adhere to the following specifications:
                - if using `ACORE` or `BFF`, must be a tuple of arrays or tensors (Y, theta, X) in this order as described by Algorithm 3 in https://arxiv.org/abs/2107.03920.
                - if using `Waldo`, must be a tuple of arrays or tensors (theta, X) in this order as described by Algorithm 1 in https://arxiv.org/pdf/2205.15680.pdf.
                - if using a custom test statistic, then an arbitrary tuple of arrays of tensors is expected.
            If not given, must supply a `simulator`.
        T_prime: Tuple[Union[np.ndarray, torch.Tensor]], optional
            Simulated dataset to train the quantile regressor to estimate critical values. Must be a tuple of arrays or tensors (theta, X).
            If not given, must supply a `simulator`.
        simulator: Simulator, optional
            If `T` and `T_prime` are not given, must pass an instance of `lf2i.simulator.Simulator`.
        b : int, optional
            Number of simulations used to estimate the test statistic. Used only if `simulator` is provided.
        b_prime : int, optional
            Number of simulations used to estimate the critical values. Used only if `simulator` is provided.
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
            quantile_regressor_kwargs = { # random search over max depth and number of trees via 5-fold CV
                'cv': {
                    'n_estimators': [100, 300, 500, 700, 1000],
                    'max_depth': [1, 3, 5, 7, 10],
                }
            }  

        # estimate test statistics
        if (not self.test_statistic._check_is_trained()) or re_estimate_test_statistics:
            print('Estimating test statistic ...', flush=True)
            if simulator:
                T = simulator.simulate_for_test_statistic(size=b, estimation_method=self.test_statistic.estimation_method)
            self.test_statistic.estimate(*T)
            
        # estimate critical values. Need to re-estimate critical values if test statistic is re-estimated because evaluated test statistic changes
        if (self.quantile_regressor is None) or re_estimate_critical_values or re_estimate_test_statistics:
            print('\nEstimating critical values ...', flush=True)
            if simulator:
                parameters_cv, samples_cv = simulator.simulate_for_critical_values(size=b_prime)
            else:
                parameters_cv, samples_cv = T_prime[0], T_prime[1]
            test_statistics_cv = self.test_statistic.evaluate(parameters_cv, samples_cv, mode='critical_values')
            self.quantile_regressor = train_qr_algorithm(
                test_statistics=test_statistics_cv,
                parameters=parameters_cv,
                algorithm=quantile_regressor,
                algorithm_kwargs=quantile_regressor_kwargs,
                alpha=confidence_level if self.test_statistic.acceptance_region == 'left' else 1-confidence_level,
                param_dim=parameters_cv.shape[1]
            )

        # construct confidence_regions
        print('\nConstructing confidence regions ...', flush=True)
        test_statistics_x = self.test_statistic.evaluate(evaluation_grid, x, mode='confidence_sets')
        confidence_regions = compute_confidence_regions(
            test_statistic=test_statistics_x,
            critical_values=self.quantile_regressor.predict(X=preprocess_predict_quantile_regression(evaluation_grid, self.quantile_regressor, self.test_statistic.poi_dim)),
            parameter_grid=to_np_if_torch(evaluation_grid),
            acceptance_region=self.test_statistic.acceptance_region,
            poi_dim=self.test_statistic.poi_dim
        )
        return confidence_regions

    def diagnostics(
        self,
        region_type: str,
        coverage_estimator: str = 'splines',
        coverage_estimator_kwargs: Dict = {},
        T_double_prime: Optional[Tuple[Union[np.ndarray, torch.Tensor]]] = None,
        simulator: Optional[Simulator] = None,
        b_double_prime : Optional[int] = None,
        new_parameters: Optional[np.ndarray] = None,
        indicators: Optional[np.ndarray] = None,
        parameters: Optional[np.ndarray] = None,
        posterior_estimator: Optional[Any] = None,
        evaluation_grid: Union[np.ndarray, torch.Tensor] = None,
        confidence_level: Optional[float] = None,
        num_p_levels: Optional[int] = 10_000
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Independent diagnostics check for the empirical coverage of a desired uncertainty quantification method across the whole parameter space.
        It estimates the coverage probability at all parameter values and provides 2-sigma prediction intervals around these estimates.
        
        NOTE: this can be applied to *any* parameter region, even if it has not been constructed via LF2I.

        Parameters
        ----------
        region_type : Union[str, None]
            Whether the parameter regions to be checked are confidence regions from 
                - LF2I ('lf2i');
                - credible regions from a posterior distribution ('posterior');
                - Gaussian prediction intervals centered around predictions ('prediction'). For this, `self.test_statistic` must be Waldo.
            If none of the above, then must provide `indicators` and `parameters`.
        coverage_estimator : str, optional
            Probabilistic classifier to use to estimate coverage probabilities, by default 'splines'
        coverage_estimator_kwargs : Dict, optional
            Settings for the probabilistic classifier, by default {}
        T_double_prime: Tuple[Union[np.ndarray, torch.Tensor]], optional
            Simulated dataset to learn the coverage probability via probabilistic classification. Must be a tuple of arrays or tensors (theta, X).
            If not given, must supply a `simulator`.
        simulator: Simulator, optional
            If `T_double_prime` is not given, must pass an instance of `lf2i.simulator.Simulator`.
        b_double_prime : int, optional
            Number of simulations used to estimate the coverage probability across the parameter space. Used only if `simulator` is provided.
        new_parameters : Optional[np.ndarray], optional
            If provided, coverage probabilities are estimated conditional on these parameters, by default None.
            If `None`, parameters simulated uniformly over the parameter space are used.
        indicators : Optional[np.ndarray], optional
            Pre-computed indicators (0-1) that mark whether the corresponding value in `parameters` is included or not in the target parameter region, by default None
        parameters : Optional[np.ndarray], optional
            Array of parameters for which the corresponding `indicators` have been pre-computed, by default None
        posterior_estimator: Any, optional
            If `region_type == posterior` and `indicators` are not provided, then a trained posterior estimator which implements the `log_prob(...)` method must be given.
        evaluation_grid: Union[np.ndarray, torch.Tensor]
            If `region_type in [`posterior`, `prediction`]` and `indicators` are not provided, grid of points over the parameter space over which to construct a 
            high-posterior-density credible region or a Gaussian interval centered around predictions.
        confidence_level : float
            If `region_type in [`posterior`, `prediction`]` and `indicators` are not provided, must give the confidence level to construct credible regions or 
            prediction intervals and compute indicators. If `region_type == `lf2i`, this information is already embedded in `self.quantile_regressor`. Must be in (0, 1).
        num_p_levels: int, optional
            If `region_type == posterior` and `indicators` are not provided, number of level sets to consider to construct the high-posterior-density credible region, by default 100_000.

        Returns
        -------
        Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Diagnostics estimator, Evaluated parameters and estimated conditional coverage probabilities (mean, upper 2-sigma bound, lower 2-sigma bound)

        Raises
        ------
        ValueError
            If `region_type` is not among those supported and `indicators is None`
        """
        if indicators is None:
            if simulator:
                parameters, samples = simulator.simulate_for_diagnostics(size=b_double_prime)
            else:
                parameters, samples = T_double_prime[0], T_double_prime[1]
        
            if region_type == 'lf2i':
                indicators = compute_indicators_lf2i(
                    test_statistics=self.test_statistic.evaluate(parameters, samples, mode='diagnostics'),
                    critical_values=self.quantile_regressor.predict(X=preprocess_predict_quantile_regression(parameters, self.quantile_regressor, parameters.shape[1])),
                    parameters=parameters,
                    acceptance_region=self.test_statistic.acceptance_region,
                    param_dim=parameters.shape[1]
                )
            elif region_type == 'posterior':
                indicators = compute_indicators_posterior(
                    posterior=posterior_estimator,
                    parameters=parameters,  # TODO: what if we want to do diagnostics against both POIs and nuisances?
                    samples=samples,
                    parameter_grid=evaluation_grid,
                    confidence_level=confidence_level,
                    param_dim=evaluation_grid.shape[1],
                    batch_size=self.test_statistic.batch_size if hasattr(self.test_statistic, "batch_size") else 1,
                    num_p_levels=num_p_levels
                )
            elif region_type == 'prediction':
                assert isinstance(self.test_statistic, Waldo), \
                        "Test statistic is not an instance of `Waldo`. You must provide `indicators` and `parameters` to diagnose prediction sets."
                indicators = compute_indicators_prediction(
                    test_statistic=self.test_statistic,
                    parameters=parameters,  # TODO: what if we want to do diagnostics against both POIs and nuisances?
                    samples=samples,
                    confidence_level=confidence_level,
                    param_dim=evaluation_grid.shape[1]
                )
            else:
                raise ValueError(
                    """If the parameter regions you want to diagnose are not from LF2I, nor they are posterior credible regions or\n 
                    gaussian prediction intervals, then you must provide `indicators` and `parameters`"""
                )
        
        diagnostics_estimator, out_parameters, mean_proba, upper_proba, lower_proba = estimate_coverage_proba(
            indicators=indicators,
            parameters=parameters,
            estimator=coverage_estimator,
            estimator_kwargs=coverage_estimator_kwargs,
            param_dim=parameters.shape[1],
            new_parameters=new_parameters
        )
        return diagnostics_estimator, out_parameters, mean_proba, upper_proba, lower_proba

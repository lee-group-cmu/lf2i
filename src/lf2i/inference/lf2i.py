from typing import Optional, Union, Dict, List, Tuple, Any, Sequence

import numpy as np
import torch

from lf2i.simulator import Simulator
from lf2i.test_statistics import TestStatistic, ACORE, BFF, Waldo
from lf2i.calibration.critical_values import train_qr_algorithm
from lf2i.calibration.p_values import augment_calibration_set, estimate_rejection_proba
from lf2i.confidence_regions.neyman_inversion import compute_confidence_regions
from lf2i.diagnostics.coverage_probability import (
    estimate_coverage_proba, 
    compute_indicators_lf2i, 
    compute_indicators_posterior, 
    compute_indicators_prediction
)
from lf2i.utils.calibration_diagnostics_inputs import preprocess_predict_quantile_regression, preprocess_predict_p_values
from lf2i.utils.miscellanea import to_np_if_torch, to_torch_if_np


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
        self.calibration_model = {}

    def inference(
        self,
        x: Union[np.ndarray, torch.Tensor],
        evaluation_grid: Union[np.ndarray, torch.Tensor],
        confidence_level: Union[float, Sequence[float]],
        calibration_method: str,
        calibration_model: Union[str, Any] = 'cat-gb',
        calibration_model_kwargs: Dict = {},
        T: Optional[Tuple[Union[np.ndarray, torch.Tensor]]] = None,
        T_prime: Optional[Tuple[Union[np.ndarray, torch.Tensor]]] = None,
        simulator: Optional[Simulator] = None,
        b: Optional[int] = None, 
        b_prime: Optional[int] = None,
        num_augment: int = 5,
        retrain_calibration: bool = False,
        verbose: bool = True
    ) -> Union[List[np.ndarray], Dict[str, List[np.ndarray]]]:
        """Estimate test statistic and critical values, and construct a confidence region for all observations in `x`.

        Parameters
        ----------
        x : Union[np.ndarray, torch.Tensor]
            Observed sample(s).
        evaluation_grid: Union[np.ndarray, torch.Tensor]
            Grid of points over the parameter space over which to invert hypothesis tests and construct the confidence regions. 
            Each confidence set will be a subset of this grid.
        confidence_level : Union[float, Sequence[float]]
            Desired confidence level(s), must be in :math:`(0, 1)`.
        calibration_method : str
            Either `critical-values` (via quantile regression) or `p-values` (via monotonic probabilistic classification).
        calibration_model : Union[str, Any], optional
            If `str`, it is an identifier for the model used for calibration, by default 'cat-gb'.
            If `Any`, must be an object implementing the `.fit(X=..., y=...)` and `.predict(X=...)` methods.
            Currently available: ['cat-gb', 'nn'] or a pre-instantiated object.
        calibration_model_kwargs : Dict, optional
            Settings for the chosen calibration model, by default {}. See modules in `lf2i.calibration` for more details.
        T: Tuple[Union[np.ndarray, torch.Tensor]], optional
            Simulated dataset to train the estimator for the test statistic. Must adhere to the following specifications:
                - if using `ACORE` or `BFF`, must be a tuple of arrays or tensors :math:`(Y, \theta, X)` in this order as described by Algorithm 3 in https://arxiv.org/abs/2107.03920.
                - if using `Waldo`, must be a tuple of arrays or tensors :math:`(\theta, X)` in this order as described by Algorithm 1 in https://arxiv.org/pdf/2205.15680.pdf.
                - if using a custom test statistic, then an arbitrary tuple of arrays of tensors is expected.
            If not given, must supply a `simulator`.
        T_prime: Tuple[Union[np.ndarray, torch.Tensor]], optional
            Simulated dataset to train the calibration model to estimate critical values or p-values. Must be a tuple of arrays or tensors :math:`(\theta, X)`.
            If not given, must supply a `simulator`.
        simulator: Simulator, optional
            If `T` and `T_prime` are not given, must pass an instance of `lf2i.simulator.Simulator`.
        b : int, optional
            Number of simulations used to estimate the test statistic. Used only if `simulator` is provided.
        b_prime : int, optional
            Number of simulations used to estimate the critical values. Used only if `simulator` is provided.
        num_augment : int
            If `calibration_method = p-values', indicates the number of cutoffs to resample for each value in `test_statistics`. 
            The augmented calibration set will be of size `num_augment` :math:`\times B^\prime`, where :math:`B^\prime` is the size of the original calibration set.
        retrain_calibration: bool, optional
            Whether to retrain the calibration model or not, even at a previously done confidence level. 
        verbose: bool, optional
            Whether to print checkpoints and progress bars or not, by default True.

        Returns
        -------
        Union[List[np.ndarray], List[List[np.ndarray]]]
            If `confidence_level` is a single value, the `i`-th element is a confidence region for the `i`-th sample in `x`.
            If `confidence_level` is a sequence of values, the `j`-th element is a list containing the confidence regions (indexed as above) at the `j`-th confidence level.
        """
        assert calibration_method in ['critical-values', 'p-values']
        self.test_statistic.verbose = verbose  # lf2i verbosity takes precedence
        
        # estimate test statistics
        if not self.test_statistic._check_is_trained():
            if verbose:
                print('Estimating test statistic ...', flush=True)
            if simulator:
                T = simulator.simulate_for_test_statistic(size=b, estimation_method=self.test_statistic.estimation_method)
            self.test_statistic.estimate(*T)  # TODO: control verbosity when estimating
            
        # estimate critical values or p-values
        if not self.calibration_model:  # need to evaluate test statistic for calibration only the first time the procedure is run
            if verbose:
                print('\nCalibration ...', flush=True)
            # save parameters and test statistics for calibration to use them for future runs with different confidence levels
            if simulator:
                # TODO: change methods name in simulators and test statistics -> calibration, no critical values
                self.parameters_calib, samples_calib = simulator.simulate_for_critical_values(size=b_prime)
            else:
                self.parameters_calib, samples_calib = T_prime[0], T_prime[1]
            self.test_statistics_calib = self.test_statistic.evaluate(self.parameters_calib, samples_calib, mode='critical_values')
        
        # TODO: calib_dict_key is necessary if training multiple quantile regressors separately at different levels alpha.
        # Eventually it should be removed because 
        #   1) no guarantee to avoid quantile crossings with separate estimation; 
        #   2) better and cheaper to estimate quantiles jointly anyway (although still no guarantee of avoiding crossings)
        calib_dict_key = f'{confidence_level:.2f}' if isinstance(confidence_level, float) else 'multiple_levels'
        if (calib_dict_key not in self.calibration_model) or retrain_calibration:
            if (calibration_model == 'cat-gb') and (calibration_model_kwargs == {}):
                self.calibration_model_kwargs = { # random search over max depth and number of trees via 5-fold CV
                    'cv': {'iterations': [100, 300, 500, 700, 1000], 'depth': [1, 3, 5, 7, 10]},
                    'n_iter': 25
                }
            else:
                self.calibration_model_kwargs = calibration_model_kwargs

            if calibration_method == 'critical-values':
                if isinstance(confidence_level, float):
                    alpha = confidence_level if self.test_statistic.acceptance_region == 'left' else 1-confidence_level
                else:
                    alpha = [cl if self.test_statistic.acceptance_region == 'left' else 1-cl for cl in confidence_level]
                
                self.calibration_model[calib_dict_key] = train_qr_algorithm(
                    test_statistics=self.test_statistics_calib,
                    parameters=self.parameters_calib,
                    algorithm=calibration_model,
                    algorithm_kwargs=self.calibration_model_kwargs,
                    alpha=alpha,
                    param_dim=self.parameters_calib.shape[1] if self.parameters_calib.ndim > 1 else 1,
                    verbose=verbose,
                    n_jobs=self.test_statistic.n_jobs if hasattr(self.test_statistic, 'n_jobs') else -2  # all cores minus 1
                )
            else:
                augmented_inputs, rejection_indicators = augment_calibration_set(
                    test_statistics=self.test_statistics_calib,
                    poi=self.parameters_calib,
                    num_augment=num_augment,
                    acceptance_region=self.test_statistic.acceptance_region
                )
                self.calibration_model[calib_dict_key] = estimate_rejection_proba(
                    inputs=augmented_inputs,
                    rejection_indicators=rejection_indicators,
                    algorithm=calibration_model,
                    algorithm_kwargs=self.calibration_model_kwargs,
                    verbose=verbose,
                    n_jobs=self.test_statistic.n_jobs if hasattr(self.test_statistic, 'n_jobs') else -2
                )

        # construct confidence_regions
        if verbose:
            print('\nConstructing confidence regions ...', flush=True)
        test_statistics_x = self.test_statistic.evaluate(evaluation_grid, x, mode='confidence_sets')
        if calibration_method == 'critical_values':
            # TODO: what if multiple levels? Do we allow this when using critical values?
            critical_values = to_np_if_torch(self.calibration_model[calib_dict_key].predict(
                X=preprocess_predict_quantile_regression(evaluation_grid, self.calibration_model[calib_dict_key], self.test_statistic.poi_dim)
            ))
            p_values = None
        else:
            
            critical_values = None
            # TODO: preprocessing should be isolated
            evaluation_grid = to_np_if_torch(evaluation_grid)
            if evaluation_grid.ndim == 1:
                evaluation_grid = np.expand_dims(evaluation_grid, axis=1)
            p_values = to_np_if_torch(self.calibration_model[calib_dict_key].predict(
                X=preprocess_predict_p_values(test_statistics_x.reshape(-1, ), np.tile(evaluation_grid, reps=(test_statistics_x.shape[0], 1)), self.calibration_model[calib_dict_key])
            ).reshape(test_statistics_x.shape[0], evaluation_grid.shape[0]))

        if isinstance(confidence_level, float):
            alpha = [1-confidence_level]
        else:
            alpha = [1-cl for cl in confidence_level]
        
        confidence_regions = []
        for a in alpha:
            confidence_regions.append(compute_confidence_regions(
                calibration_method=calibration_method,
                test_statistic=test_statistics_x,
                parameter_grid=to_np_if_torch(evaluation_grid),
                critical_values=critical_values,
                p_values=p_values,
                alpha=a,
                acceptance_region=self.test_statistic.acceptance_region,
                poi_dim=self.test_statistic.poi_dim
            ))
        return confidence_regions if len(alpha) > 1 else confidence_regions[0]

    def diagnostics(
        self,
        region_type: str,
        confidence_level: float,
        calibration_method: Optional[str] = None,
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
        num_level_sets: Optional[int] = 10_000,
        norm_posterior_samples: Optional[int] = None,
        verbose: bool = True
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
        calibration_method : str, optional
            If `region_type = 'lf2i', either `critical-values` or `p-values`, ignored otherwise.
        confidence_level : float
            If `region_type in [`posterior`, `prediction`]` and `indicators` are not provided, must give the confidence level to construct credible regions or 
            prediction intervals and compute indicators. If `region_type == `lf2i`, needed to evaluate the calibration method. Must be in (0, 1).
        coverage_estimator : str, optional
            Probabilistic classifier to use to estimate coverage probabilities, by default 'splines'. Currently supported: ['splines', 'cat-gb'].
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
        num_level_sets: int, optional
            If `region_type == posterior` and `indicators` are not provided, Number of level sets to examine, by default 10_000. 
            A high number of level sets ensures the actual credible level is as close as possible to the specified one.
        norm_posterior_samples : int, optional
            Number of samples to use to estimate the leakage correction factor, by default None. More samples lead to better estimates of the normalization constant when using a normalized posterior.
            If `None`, uses the un-normalized posterior (but note that the density is already being explicitly normalized over the `evaluation_grid` to compute the HPD region).
        verbose: bool, optional
            Whether to print checkpoints and progress bars or not, by default True.
            
        Returns
        -------
        Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Diagnostics estimator, Evaluated parameters and estimated conditional coverage probabilities (mean, upper 2-sigma bound, lower 2-sigma bound)

        Raises
        ------
        ValueError
            If `region_type` is not among those supported and `indicators is None`
        """
        assert calibration_method in ['critical-values', 'p-values']
        self.test_statistic.verbose = verbose  # lf2i verbosity takes precedence
        
        if indicators is None:
            if simulator:
                parameters, samples = simulator.simulate_for_diagnostics(size=b_double_prime)
            else:
                parameters, samples = T_double_prime[0], T_double_prime[1]
        
            if region_type == 'lf2i':
                calib_dict_key = f'{confidence_level:.2f}' if isinstance(confidence_level, float) else 'multiple_levels'
                test_statistics = self.test_statistic.evaluate(parameters, samples, mode='diagnostics')
                if calibration_method == 'critical-values':
                    critical_values = to_np_if_torch(self.calibration_model[calib_dict_key].predict(
                        X=preprocess_predict_quantile_regression(parameters, self.calibration_model[calib_dict_key], parameters.shape[1] if parameters.ndim > 1 else 1)
                    ))
                    p_values = None
                else:
                    critical_values = None
                    p_values = to_np_if_torch(self.calibration_model[calib_dict_key].predict(
                        X=preprocess_predict_p_values(test_statistics, parameters, self.calibration_model[calib_dict_key])
                    ))

                indicators = compute_indicators_lf2i(
                    calibration_method=calibration_method,
                    test_statistics=test_statistics,
                    parameters=parameters,
                    critical_values=critical_values,
                    p_values=p_values,
                    alpha=1-confidence_level,
                    acceptance_region=self.test_statistic.acceptance_region,
                    param_dim=parameters.shape[1] if parameters.ndim > 1 else 1
                )
            elif region_type == 'posterior':
                indicators = compute_indicators_posterior(
                    posterior=posterior_estimator,
                    parameters=parameters,  # TODO: what if we want to do diagnostics against both POIs and nuisances?
                    samples=samples,
                    parameter_grid=to_torch_if_np(evaluation_grid),
                    credible_level=confidence_level,
                    param_dim=evaluation_grid.shape[1] if evaluation_grid.ndim > 1 else 1,
                    batch_size=self.test_statistic.batch_size if hasattr(self.test_statistic, "batch_size") else 1,
                    num_level_sets=num_level_sets,
                    norm_posterior_samples=norm_posterior_samples
                )
            elif region_type == 'prediction':
                assert isinstance(self.test_statistic, Waldo), \
                        "Test statistic is not an instance of `Waldo`. You must provide `indicators` and `parameters` to diagnose prediction sets."
                indicators = compute_indicators_prediction(
                    test_statistic=self.test_statistic,
                    parameters=parameters,  # TODO: what if we want to do diagnostics against both POIs and nuisances?
                    samples=samples,
                    confidence_level=confidence_level,
                    param_dim=evaluation_grid.shape[1] if evaluation_grid.ndim > 1 else 1
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
            param_dim=parameters.shape[1] if parameters.ndim > 1 else 1,
            new_parameters=new_parameters
        )
        return diagnostics_estimator, out_parameters, mean_proba, upper_proba, lower_proba

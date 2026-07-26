import gc
import warnings
from typing import Optional, Union, Dict, List, Tuple, Any, Sequence

import numpy as np
import torch

from lf2i.simulator import Simulator
from lf2i.test_statistics import TestStatistic, ACORE, BFF, Waldo, Posterior
from lf2i.calibration.critical_values import train_qr_algorithm
from lf2i.calibration.p_values import estimate_rejection_proba
from lf2i.confidence_regions.neyman_inversion import compute_confidence_regions
from lf2i.diagnostics.coverage_probability import (
    estimate_coverage_proba,
    compute_indicators_lf2i,
    compute_indicators_posterior,
    compute_indicators_prediction,
)
from lf2i.utils.calibration_diagnostics_inputs import preprocess_predict_quantile_regression, preprocess_predict_p_values
from lf2i.utils.miscellanea import to_np_if_torch, to_torch_if_np, to_np_if_pd


class LF2I:
    """
    High-level entry point to do inference with LF2I (https://arxiv.org/abs/2107.03920).
    This allows to quickly construct confidence regions for parameters of interest in an SBI setting leveraging an arbitrary estimator

    - of the *likelihood*, using for example the ACORE or BFF test statistics (https://arxiv.org/pdf/2002.10399.pdf, https://arxiv.org/abs/2107.03920);
    - of the *posterior*, using for example the Posterior (10.1088/2632-2153/ae67cd) or Waldo test statistic (https://arxiv.org/abs/2205.15680);

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
        elif test_statistic == 'posterior':
            self.test_statistic = Posterior(**test_statistic_kwargs)
        elif isinstance(test_statistic, TestStatistic):
            self.test_statistic = test_statistic
        else:
            raise ValueError(f"Expected one of `acore`, `bff`, `waldo`, `posterior`, or an instance of a custom `lf2i.test_statistics._base.TestStatistic`, got {test_statistic}")
        self.calibration_model = {}
        self.recalibrate_p_values = False
        self.parameters_calib = None

    # ------------------------------------------------------------------
    # Public: inference
    # ------------------------------------------------------------------

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
        recalibrate_p_values: bool = False,
        verbose: bool = True,
    ) -> Union[List[np.ndarray], Dict[str, List[np.ndarray]]]:
        """Estimate test statistic and critical values, and construct confidence sets for all observations in `x`.

        Parameters
        ----------
        x : Union[np.ndarray, torch.Tensor]
            Observed sample(s).
        evaluation_grid: Union[np.ndarray, torch.Tensor]
            Grid of points over the parameter space over which to invert hypothesis tests.
            Each confidence set will be a subset of this grid.
        confidence_level : Union[float, Sequence[float]]
            Desired confidence level(s), must be in :math:`(0, 1)`.
        calibration_method : str
            Either `critical-values` (via quantile regression) or `p-values` (via monotonic probabilistic classification).
        calibration_model : Union[str, Any], optional
            If `str`, identifier for the calibration model, by default 'cat-gb'.
        calibration_model_kwargs : Dict, optional
            Settings for the chosen calibration model, by default {}.
        T: Tuple[Union[np.ndarray, torch.Tensor]], optional
            Simulated dataset to train the test statistic estimator.
        T_prime: Tuple[Union[np.ndarray, torch.Tensor]], optional
            Simulated dataset to train the calibration model.
        simulator: Simulator, optional
            If `T` and `T_prime` are not given, must pass an instance of `lf2i.simulator.Simulator`.
        b : int, optional
            Number of simulations for the test statistic. Used only if `simulator` is provided.
        b_prime : int, optional
            Number of simulations for calibration. Used only if `simulator` is provided.
        num_augment : int
            If `calibration_method = 'p-values'`, number of cutoffs to resample per value.
        retrain_calibration: bool, optional
            Whether to retrain the calibration model, by default False.
        recalibrate_p_values: bool, optional
            Whether to hold out part of the calibration set to recalibrate p-value thresholds, by default False.
        verbose: bool, optional
            Whether to print checkpoints and progress bars, by default True.

        Returns
        -------
        List[np.ndarray] or List[List[np.ndarray]] for multiple confidence levels.
        """
        assert calibration_method in ['critical-values', 'p-values']

        self.test_statistic.verbose = verbose
        self.recalibrate_p_values = recalibrate_p_values or False

        # --- reshape x to (n_obs, param_dim) if needed ---
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # --- train test statistic ---
        if not self.test_statistic._check_is_trained():
            if verbose:
                print('Estimating test statistic ...', flush=True)
            if simulator:
                T = simulator.simulate_for_test_statistic(size=b, estimation_method=self.test_statistic.estimation_method)
            self.test_statistic.estimate(*T)

        # --- train calibration model ---
        calibration_model_is_prebuilt = isinstance(calibration_model, dict)
        if calibration_model is not None and calibration_model_is_prebuilt:
            self.calibration_model = calibration_model
            if T_prime is not None:
                self.parameters_calib, self.samples_calib = T_prime[0], T_prime[1]
                self.test_statistics_calib = self.test_statistic.evaluate(self.parameters_calib, self.samples_calib, mode='critical_values')

        if not self.calibration_model:
            if verbose:
                print('\nCalibration ...', flush=True)
            if simulator:
                self.parameters_calib, self.samples_calib = simulator.simulate_for_critical_values(size=b_prime)
            else:
                self.parameters_calib, self.samples_calib = T_prime[0], T_prime[1]
            self.test_statistics_calib = self.test_statistic.evaluate(self.parameters_calib, self.samples_calib, mode='critical_values')
        else:
            if verbose:
                print('\nCalibration already complete', flush=True)

        # --- optional p-value recalibration holdout ---
        if calibration_method == 'p-values' and recalibrate_p_values:
            holdout_set_size = min(1000, len(self.parameters_calib) // 10)
            self.holdout_parameters_calib, self.holdout_samples_calib, self.holdout_test_statistics_calib = (
                self.parameters_calib[-holdout_set_size:],
                self.samples_calib[-holdout_set_size:],
                self.test_statistics_calib[-holdout_set_size:],
            )
            self.parameters_calib, self.samples_calib, self.test_statistics_calib = (
                self.parameters_calib[:-holdout_set_size],
                self.samples_calib[:-holdout_set_size],
                self.test_statistics_calib[:-holdout_set_size],
            )
        else:
            self.holdout_parameters_calib, self.holdout_samples_calib, self.holdout_test_statistics_calib = None, None, None

        calib_dict_key = f'{confidence_level:.2f}' if isinstance(confidence_level, float) else 'multiple_levels'
        if calibration_model_is_prebuilt and (calib_dict_key not in self.calibration_model or retrain_calibration):
            raise KeyError(
                f"calibration_model was passed as a pre-built dict, but it has no entry for "
                f"confidence level key '{calib_dict_key}' (available keys: {list(self.calibration_model.keys())}) "
                f"or retrain_calibration=True was requested. A pre-built dict cannot be used as a training "
                f"algorithm identifier; pass a dict that already contains this key, or pass calibration_model "
                f"as a str/algorithm instance to train a new calibration model."
            )
        if (calib_dict_key not in self.calibration_model) or retrain_calibration:
            if verbose:
                print('\nRetraining calibration...')
            if (calibration_model == 'cat-gb') and (calibration_model_kwargs == {}):
                self.calibration_model_kwargs = {
                    'cv': {'iterations': [100, 300, 500, 700, 1000], 'depth': [1, 3, 5, 7, 10]},
                    'n_iter': 25
                }
            else:
                self.calibration_model_kwargs = calibration_model_kwargs

            if calibration_method == 'critical-values':
                if isinstance(confidence_level, float):
                    alpha = confidence_level if self.test_statistic.acceptance_region == 'left' else 1 - confidence_level
                else:
                    alpha = [cl if self.test_statistic.acceptance_region == 'left' else 1 - cl for cl in confidence_level]
                self.calibration_model[calib_dict_key] = train_qr_algorithm(
                    test_statistics=self.test_statistics_calib,
                    parameters=self.parameters_calib,
                    algorithm=calibration_model,
                    algorithm_kwargs=self.calibration_model_kwargs,
                    alpha=alpha,
                    param_dim=self.parameters_calib.shape[1] if self.parameters_calib.ndim > 1 else 1,
                    verbose=verbose,
                    n_jobs=self.test_statistic.n_jobs if hasattr(self.test_statistic, 'n_jobs') else -2,
                )
            else:
                self.calibration_model[calib_dict_key] = estimate_rejection_proba(
                    test_statistics=self.test_statistics_calib,
                    parameters=self.parameters_calib,
                    algorithm=calibration_model,
                    augment_kwargs={'num_augment': num_augment},
                    verbose=verbose,
                )

        # --- evaluate test statistics over evaluation_grid ---
        if verbose:
            print('\nConstructing confidence sets ...', flush=True)
        test_statistics_x = self.test_statistic.evaluate(evaluation_grid, x, mode='confidence_sets')

        if calibration_method == 'critical-values':
            critical_values = to_np_if_pd(self.calibration_model[calib_dict_key].predict(
                preprocess_predict_quantile_regression(evaluation_grid, self.calibration_model[calib_dict_key], self.test_statistic.param_dim)
            ))
            p_values = None
        else:
            if verbose:
                print('\nComputing p-values...')
            critical_values = None
            _pv_col = 0 if self.test_statistic.acceptance_region == 'left' else 1
            p_values = self.calibration_model[calib_dict_key].predict_proba(
                X=preprocess_predict_p_values('confidence_sets', test_statistics_x, evaluation_grid, self.calibration_model[calib_dict_key])
            )[:, _pv_col]

        # --- recalibrate alpha ---
        alpha_list = [1 - confidence_level] if isinstance(confidence_level, float) else [1 - cl for cl in confidence_level]
        if (
            self.holdout_parameters_calib is not None
            and self.holdout_test_statistics_calib is not None
            and self.holdout_samples_calib is not None
            and calibration_method == 'p-values'
        ):
            if verbose:
                print('\nRe-calibrating p-values on holdout set ...', flush=True)
            self.holdout_p_values = self.calibration_model[calib_dict_key].predict_proba(
                X=preprocess_predict_p_values('holdout_calibration', self.holdout_test_statistics_calib, self.holdout_parameters_calib, self.calibration_model[calib_dict_key])
            )[:, _pv_col]
            alpha_list = [np.quantile(self.holdout_p_values, a) for a in alpha_list]
            if verbose:
                for cl, a in zip(
                    [confidence_level] if isinstance(confidence_level, float) else confidence_level,
                    alpha_list,
                ):
                    print(f'Original alpha: {1 - cl}, Re-calibrated alpha: {a}')
        else:
            self.holdout_p_values = None

        return self._construct_confidence_regions(
            calibration_method=calibration_method,
            test_statistics_x=test_statistics_x,
            evaluation_grid=evaluation_grid,
            critical_values=critical_values,
            p_values=p_values,
            alpha_list=alpha_list,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Private: inference dispatch helpers
    # ------------------------------------------------------------------

    def _construct_confidence_regions(
        self,
        calibration_method: str,
        test_statistics_x,
        evaluation_grid,
        critical_values,
        p_values,
        alpha_list: List[float],
        verbose: bool = False,
    ) -> Union[List[np.ndarray], List[List[np.ndarray]]]:
        if critical_values is not None:
            assert critical_values.size == len(to_np_if_torch(evaluation_grid)) * len(alpha_list), \
                f"critical_values has {critical_values.size} elements, expected {len(to_np_if_torch(evaluation_grid)) * len(alpha_list)} (grid_size x n_levels)"

        confidence_regions = []
        for idx, a in enumerate(alpha_list):
            if verbose:
                print(f'\nCreating set {idx}...')
            confidence_regions.append(compute_confidence_regions(
                calibration_method=calibration_method,
                test_statistic=test_statistics_x,
                parameter_grid=evaluation_grid,
                critical_values=critical_values.reshape(-1, len(alpha_list))[:, idx] if critical_values is not None else None,
                p_values=p_values,
                alpha=a,
                acceptance_region=self.test_statistic.acceptance_region,
                poi_dim=self.test_statistic.param_dim,
            ))
        return confidence_regions if len(alpha_list) > 1 else confidence_regions[0]


    # ------------------------------------------------------------------
    # Public: coverage
    # ------------------------------------------------------------------

    def coverage(
        self,
        region_type: str,
        confidence_level: float,
        calibration_method: Optional[str] = None,
        coverage_estimator: str = 'cat-gb',
        coverage_estimator_kwargs: Dict = {},
        T_double_prime: Optional[Tuple[Union[np.ndarray, torch.Tensor]]] = None,
        simulator: Optional[Simulator] = None,
        b_double_prime: Optional[int] = None,
        new_parameters: Optional[np.ndarray] = None,
        indicators: Optional[np.ndarray] = None,
        parameters: Optional[np.ndarray] = None,
        posterior_estimator: Optional[Any] = None,
        evaluation_grid: Union[np.ndarray, torch.Tensor] = None,
        num_level_sets: Optional[int] = 10_000,
        n_jobs: Optional[int] = -2,
        verbose: bool = True,
        exact: Optional[bool] = None,
        monte_carlo_size: int = 500,
        parameter_grid: torch.Tensor = None,
        **posterior_kwargs,
    ):
        """Estimate or compute exactly the coverage probability of a confidence/credible region method.

        Dispatches to :meth:`_estimated_coverage` (ML-based estimator) or
        :meth:`_exact_coverage` (Monte Carlo), depending on ``exact`` and whether a
        simulator is available. Only parameters specific to this dispatch are documented
        below; for all other parameters, see :meth:`diagnostics`.

        Parameters
        ----------
        exact : bool, optional
            If True, use Monte Carlo exact coverage (requires ``simulator``).
            If False, use the ML estimator (requires ``T_double_prime`` or ``simulator``).
            If None (default), use MC when a ``simulator`` is provided, otherwise estimated.
        monte_carlo_size : int, optional
            MC draws per grid point for ``_exact_coverage``. Default 500.
        parameter_grid : torch.Tensor, optional
            Dense parameter grid for ``region_type='posterior'`` exact coverage.
        """
        if exact is None:
            exact = simulator is not None and (indicators is None)

        if exact:
            return self._exact_coverage(
                region_type=region_type,
                confidence_level=confidence_level,
                calibration_method=calibration_method,
                simulator=simulator,
                evaluation_grid=evaluation_grid,
                monte_carlo_size=monte_carlo_size,
                posterior_estimator=posterior_estimator,
                parameter_grid=parameter_grid,
                num_level_sets=num_level_sets,
                n_jobs=n_jobs,
                **posterior_kwargs,
            )
        else:
            return self._estimated_coverage(
                region_type=region_type,
                confidence_level=confidence_level,
                calibration_method=calibration_method,
                coverage_estimator=coverage_estimator,
                coverage_estimator_kwargs=coverage_estimator_kwargs,
                T_double_prime=T_double_prime,
                simulator=simulator,
                b_double_prime=b_double_prime,
                new_parameters=new_parameters,
                indicators=indicators,
                parameters=parameters,
                posterior_estimator=posterior_estimator,
                evaluation_grid=evaluation_grid,
                num_level_sets=num_level_sets,
                n_jobs=n_jobs,
                verbose=verbose,
                **posterior_kwargs,
            )

    def _estimated_coverage(
        self,
        region_type: str,
        confidence_level: float,
        calibration_method: Optional[str] = None,
        coverage_estimator: str = 'cat-gb',
        coverage_estimator_kwargs: Dict = {},
        T_double_prime: Optional[Tuple[Union[np.ndarray, torch.Tensor]]] = None,
        simulator: Optional[Simulator] = None,
        b_double_prime: Optional[int] = None,
        new_parameters: Optional[np.ndarray] = None,
        indicators: Optional[np.ndarray] = None,
        parameters: Optional[np.ndarray] = None,
        posterior_estimator: Optional[Any] = None,
        evaluation_grid: Union[np.ndarray, torch.Tensor] = None,
        num_level_sets: Optional[int] = 10_000,
        n_jobs: Optional[int] = -2,
        verbose: bool = True,
        **posterior_kwargs,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ML-based coverage estimator across the parameter space.

        Independent diagnostics check via probabilistic classification.
        Can be applied to *any* parameter region, even if not constructed via LF2I.
        """
        sizes = None

        if region_type == 'lf2i':
            assert calibration_method in ['critical-values', 'p-values']
        self.test_statistic.verbose = verbose

        if indicators is None:
            if simulator:
                parameters, samples = simulator.simulate_for_diagnostics(size=b_double_prime)
            else:
                parameters, samples = T_double_prime[0], T_double_prime[1]

            if region_type == 'lf2i':
                calib_dict_key = f'{confidence_level:.2f}' if 'multiple_levels' not in self.calibration_model else 'multiple_levels'
                test_statistics = self.test_statistic.evaluate(parameters, samples, mode='diagnostics')
                if calibration_method == 'critical-values':
                    critical_values = to_np_if_torch(to_np_if_pd(self.calibration_model[calib_dict_key].predict(
                        preprocess_predict_quantile_regression(parameters, self.calibration_model[calib_dict_key], parameters.shape[1] if parameters.ndim > 1 else 1)
                    )))
                    p_values = None
                    if calib_dict_key == 'multiple_levels':
                        idx_cl = np.argmin(np.abs(
                            confidence_level - (1 - np.array(self.calibration_model['multiple_levels'].estimator.get_params()['loss_function'].split('=')[1].split(',')).astype(float))
                        ))
                        critical_values = critical_values[:, idx_cl]
                else:
                    critical_values = None
                    _pv_col = 0 if self.test_statistic.acceptance_region == 'left' else 1
                    p_values = to_np_if_torch(self.calibration_model[calib_dict_key].predict_proba(
                        X=preprocess_predict_p_values('diagnostics', test_statistics, parameters, self.calibration_model[calib_dict_key])
                    )[:, _pv_col])

                if calibration_method == 'p-values' and self.recalibrate_p_values and self.holdout_parameters_calib is not None and self.holdout_test_statistics_calib is not None and self.holdout_samples_calib is not None:
                    if verbose:
                        print('\nRe-calibrating p-values on holdout set ...', flush=True)
                    alpha = np.quantile(self.holdout_p_values, 1 - confidence_level)
                    if verbose:
                        print(f'Original alpha: {1 - confidence_level}, Re-calibrated alpha: {alpha}')
                else:
                    alpha = 1 - confidence_level

                indicators = compute_indicators_lf2i(
                    calibration_method=calibration_method,
                    test_statistics=test_statistics,
                    parameters=parameters,
                    critical_values=critical_values,
                    p_values=p_values,
                    alpha=alpha,
                    acceptance_region=self.test_statistic.acceptance_region,
                    param_dim=parameters.shape[1] if parameters.ndim > 1 else 1,
                )

            elif region_type == 'posterior':
                if not posterior_kwargs:
                    if hasattr(self.test_statistic, 'posterior_kwargs'):
                        posterior_kwargs = self.test_statistic.posterior_kwargs
                    else:
                        posterior_kwargs = {}
                indicators, sizes = compute_indicators_posterior(
                    posterior=posterior_estimator,
                    parameters=parameters,
                    samples=samples,
                    parameter_grid=to_torch_if_np(evaluation_grid),
                    credible_level=confidence_level,
                    param_dim=evaluation_grid.shape[1] if evaluation_grid.ndim > 1 else 1,
                    batch_size=self.test_statistic.batch_size if hasattr(self.test_statistic, "batch_size") else 1,
                    num_level_sets=num_level_sets,
                    n_jobs=n_jobs,
                    return_size=True,
                    **posterior_kwargs,
                )

            elif region_type == 'prediction':
                assert isinstance(self.test_statistic, Waldo), \
                    "Test statistic is not an instance of `Waldo`. You must provide `indicators` and `parameters` to diagnose prediction sets."
                indicators = compute_indicators_prediction(
                    test_statistic=self.test_statistic,
                    parameters=parameters,
                    samples=samples,
                    confidence_level=confidence_level,
                    param_dim=evaluation_grid.shape[1] if evaluation_grid.ndim > 1 else 1,
                )
            else:
                raise ValueError(
                    "If the parameter regions you want to diagnose are not from LF2I, nor they are posterior credible regions or\n "
                    "gaussian prediction intervals, then you must provide `indicators` and `parameters`"
                )

        diagnostics_estimator, out_parameters, mean_proba, upper_proba, lower_proba = estimate_coverage_proba(
            indicators=indicators,
            parameters=parameters,
            estimator=coverage_estimator,
            estimator_kwargs=coverage_estimator_kwargs,
            param_dim=parameters.shape[1] if parameters.ndim > 1 else 1,
            new_parameters=new_parameters,
        )

        if region_type == 'posterior' and sizes is not None:
            return diagnostics_estimator, out_parameters, mean_proba, upper_proba, lower_proba, sizes
        return diagnostics_estimator, out_parameters, mean_proba, upper_proba, lower_proba

    def _exact_coverage(
        self,
        region_type: str,
        confidence_level: Union[float, Sequence[float]],
        calibration_method: Optional[str] = None,
        simulator: Optional[Simulator] = None,
        evaluation_grid: Optional[np.ndarray] = None,
        monte_carlo_size: int = 500,
        posterior_estimator: Optional[Any] = None,
        parameter_grid: Optional[torch.Tensor] = None,
        num_level_sets: int = 10_000,
        n_jobs: int = -2,
        **posterior_kwargs,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, Dict[float, np.ndarray]]]:
        """Monte Carlo exact coverage at each point of ``evaluation_grid``.

        Parameters
        ----------
        simulator : Simulator
            lf2i Simulator used to draw samples.
        evaluation_grid : np.ndarray, shape (n_grid, param_dim)
            Grid of parameter values at which to evaluate coverage.
        confidence_level : Union[float, Sequence[float]]
            Nominal confidence level(s), each in (0, 1).
        calibration_method : str, optional
            Either 'critical-values' or 'p-values'. Required when ``region_type='lf2i'``.
        monte_carlo_size : int, optional
            Number of MC draws per grid point. Default 500.
        region_type : str, optional
            ``'lf2i'`` uses the calibration model; ``'posterior'`` evaluates HPD credible
            region coverage.
        posterior_estimator : optional
            Trained posterior with a ``log_prob`` method. Required when ``region_type='posterior'``.
        parameter_grid : torch.Tensor, optional
            Dense grid for HPD approximation. Required when ``region_type='posterior'``.
        num_level_sets : int, optional
            HPD binary-search resolution. Default 10_000.
        n_jobs : int, optional
            Joblib parallelism for HPD computation. Default -2.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(evaluation_grid, coverage_per_grid_point)`` if ``confidence_level`` is scalar.
        Tuple[np.ndarray, Dict[float, np.ndarray]]
            ``(evaluation_grid, {cl: coverage_per_grid_point, ...})`` if a sequence.
        """
        if region_type == 'posterior':
            if posterior_estimator is None or parameter_grid is None:
                raise ValueError(
                    "region_type='posterior' requires both `posterior_estimator` and `parameter_grid`."
                )
            from lf2i.diagnostics.monte_carlo_methods import monte_carlo_coverage_posterior
            return monte_carlo_coverage_posterior(
                posterior_estimator=posterior_estimator,
                simulator=simulator,
                evaluation_grid=evaluation_grid,
                credible_level=confidence_level,
                parameter_grid=parameter_grid,
                monte_carlo_size=monte_carlo_size,
                num_level_sets=num_level_sets,
                n_jobs=n_jobs,
                **posterior_kwargs,
            )
        if calibration_method is None:
            raise ValueError("`calibration_method` is required when region_type='lf2i'.")
        from lf2i.diagnostics.monte_carlo_methods import monte_carlo_coverage
        return monte_carlo_coverage(
            test_statistic=self.test_statistic,
            calibration_model=self.calibration_model,
            simulator=simulator,
            evaluation_grid=evaluation_grid,
            confidence_level=confidence_level,
            calibration_method=calibration_method,
            monte_carlo_size=monte_carlo_size,
        )

    def diagnostics(
        self,
        region_type: str,
        confidence_level: float,
        calibration_method: Optional[str] = None,
        coverage_estimator: str = 'cat-gb',
        coverage_estimator_kwargs: Dict = {},
        T_double_prime: Optional[Tuple[Union[np.ndarray, torch.Tensor]]] = None,
        simulator: Optional[Simulator] = None,
        b_double_prime: Optional[int] = None,
        new_parameters: Optional[np.ndarray] = None,
        indicators: Optional[np.ndarray] = None,
        parameters: Optional[np.ndarray] = None,
        posterior_estimator: Optional[Any] = None,
        evaluation_grid: Union[np.ndarray, torch.Tensor] = None,
        num_level_sets: Optional[int] = 10_000,
        n_jobs: Optional[int] = -2,
        verbose: bool = True,
        **posterior_kwargs,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ML-based coverage diagnostics. Deprecated alias for :meth:`_estimated_coverage`.

        .. deprecated::
            ``diagnostics`` will be removed in a future release.  Use :meth:`coverage` instead.
        """
        warnings.warn(
            "LF2I.diagnostics() is deprecated and will be removed in a future release. "
            "Use LF2I.coverage() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._estimated_coverage(
            region_type=region_type,
            confidence_level=confidence_level,
            calibration_method=calibration_method,
            coverage_estimator=coverage_estimator,
            coverage_estimator_kwargs=coverage_estimator_kwargs,
            T_double_prime=T_double_prime,
            simulator=simulator,
            b_double_prime=b_double_prime,
            new_parameters=new_parameters,
            indicators=indicators,
            parameters=parameters,
            posterior_estimator=posterior_estimator,
            evaluation_grid=evaluation_grid,
            num_level_sets=num_level_sets,
            n_jobs=n_jobs,
            verbose=verbose,
            **posterior_kwargs,
        )

    def mc_diagnostics(
        self,
        simulator,
        evaluation_grid: np.ndarray,
        confidence_level: Union[float, Sequence[float]],
        calibration_method: str = None,
        monte_carlo_size: int = 500,
        region_type: str = 'lf2i',
        posterior_estimator=None,
        parameter_grid: torch.Tensor = None,
        num_level_sets: int = 10_000,
        n_jobs: int = -2,
        **posterior_kwargs,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, Dict[float, np.ndarray]]]:
        """MC-exact coverage diagnostics. Deprecated alias for :meth:`_exact_coverage`.

        .. deprecated::
            ``mc_diagnostics`` will be removed in a future release.  Use :meth:`coverage` instead.
        """
        warnings.warn(
            "LF2I.mc_diagnostics() is deprecated and will be removed in a future release. "
            "Use LF2I.coverage(exact=True) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._exact_coverage(
            region_type=region_type,
            confidence_level=confidence_level,
            calibration_method=calibration_method,
            simulator=simulator,
            evaluation_grid=evaluation_grid,
            monte_carlo_size=monte_carlo_size,
            posterior_estimator=posterior_estimator,
            parameter_grid=parameter_grid,
            num_level_sets=num_level_sets,
            n_jobs=n_jobs,
            **posterior_kwargs,
        )

    # ------------------------------------------------------------------
    # Public: power
    # ------------------------------------------------------------------

    def power(
        self,
        evaluation_grid: Union[np.ndarray, torch.Tensor],
        confidence_level: Union[float, Sequence[float]],
        T_double_prime: Optional[Tuple[Union[np.ndarray, torch.Tensor]]] = None,
        simulator: Optional[Simulator] = None,
        calibration_method: str = 'critical-values',
        batch_size: int = 1000,
        monte_carlo_size: int = 100,
        exact: Optional[bool] = None,
        verbose: bool = True,
    ):
        """Estimate or compute exactly the power (expected confidence set size) of the LF2I procedure.

        Dispatches to :meth:`_estimated_power` (regression-based) or :meth:`_exact_power`
        (Monte Carlo), depending on ``exact`` and whether a simulator is available.

        Parameters
        ----------
        evaluation_grid : Union[np.ndarray, torch.Tensor]
            Grid of parameter values at which to evaluate power.
        confidence_level : Union[float, Sequence[float]]
            Nominal confidence level(s), each in (0, 1).
        T_double_prime : Tuple, optional
            Pre-simulated dataset ``(parameters, samples)`` for estimated power.
        simulator : Simulator, optional
            Required for exact (MC) power.
        calibration_method : str, optional
            ``'critical-values'`` or ``'p-values'``. Default ``'critical-values'``.
        batch_size : int, optional
            Batch size for estimated power. Default 1000.
        monte_carlo_size : int, optional
            MC draws per grid point for exact power. Default 100.
        exact : bool, optional
            If None (default), use MC when ``simulator`` is available, else estimated.
        verbose : bool, optional
            Whether to print progress. Default True.

        Returns
        -------
        For estimated power: ``np.ndarray`` of confidence set sizes (length = n_samples in T_double_prime).
        For exact power: ``(evaluation_grid, mean_size_per_grid_point)``.
        """
        if exact is None:
            exact = simulator is not None

        if exact:
            return self._exact_power(
                simulator=simulator,
                evaluation_grid=evaluation_grid,
                confidence_level=confidence_level,
                calibration_method=calibration_method,
                monte_carlo_size=monte_carlo_size,
                verbose=verbose,
            )
        else:
            return self._estimated_power(
                T_double_prime=T_double_prime,
                evaluation_grid=evaluation_grid,
                confidence_level=confidence_level,
                calibration_method=calibration_method,
                batch_size=batch_size,
                verbose=verbose,
            )

    def _estimated_power(
        self,
        T_double_prime: Tuple[Union[np.ndarray, torch.Tensor]],
        evaluation_grid: Union[np.ndarray, torch.Tensor],
        confidence_level: Union[float, Sequence[float]],
        calibration_method: str = 'critical-values',
        batch_size: int = 1000,
        verbose: bool = True,
    ) -> np.ndarray:
        """Regression-based power estimate: train a quantile regressor on confidence set sizes.

        Runs inference in batches on ``T_double_prime`` samples, computes confidence set
        sizes, then fits a quantile regressor predicting size from the true parameter.

        Returns
        -------
        np.ndarray
            Confidence set sizes for all samples in ``T_double_prime``.
        """
        b_double_prime_params, b_double_prime_samples = T_double_prime
        n_samples = len(b_double_prime_samples)
        n_batches = int(np.ceil(n_samples / batch_size))

        if verbose:
            print(f"Processing {n_samples} samples in {n_batches} batches of size {batch_size}")

        b_double_prime_sizes = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            if verbose:
                print(f"Batch {i + 1}/{n_batches}: samples {start_idx} to {end_idx}")

            batch_samples = b_double_prime_samples[start_idx:end_idx]

            confidence_sets_batch = self.inference(
                x=batch_samples,
                evaluation_grid=evaluation_grid,
                confidence_level=confidence_level,
                calibration_method=calibration_method,
                calibration_model=self.calibration_model,
                verbose=False,
            )

            if not isinstance(confidence_level, float) and len(confidence_level) > 1:
                confidence_sets_batch = confidence_sets_batch[0]

            batch_sizes = np.array([
                cs.shape[0] / evaluation_grid.shape[0]
                for cs in confidence_sets_batch
            ])
            b_double_prime_sizes.append(batch_sizes)

            del confidence_sets_batch, batch_samples, batch_sizes
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        b_double_prime_sizes = np.concatenate(b_double_prime_sizes)

        if verbose:
            print(f"Training power model on {len(b_double_prime_sizes)} samples...")

        self.power_model = train_qr_algorithm(
            test_statistics=b_double_prime_sizes,
            parameters=b_double_prime_params,
            algorithm='cat-gb',
            algorithm_kwargs={'iterations': 100, 'depth': 3},
            alpha=0.5,
            param_dim=self.parameters_calib.shape[1] if self.parameters_calib.ndim > 1 else 1,
            verbose=verbose,
            n_jobs=self.test_statistic.n_jobs if hasattr(self.test_statistic, 'n_jobs') else -2,
        )

        return b_double_prime_sizes

    def _exact_power(
        self,
        simulator: Simulator,
        evaluation_grid: Union[np.ndarray, torch.Tensor],
        confidence_level: Union[float, Sequence[float]],
        calibration_method: str = 'critical-values',
        monte_carlo_size: int = 100,
        verbose: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, Dict[float, np.ndarray]]]:
        """MC-exact power (expected confidence set size) at each evaluation grid point.

        For each θ* in ``evaluation_grid``, draws ``monte_carlo_size`` samples from
        f(x | θ*), computes confidence sets for each, and returns the mean fraction of
        the grid covered.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(evaluation_grid_np, mean_size_per_grid_point)`` if ``confidence_level`` is scalar.
        Tuple[np.ndarray, Dict[float, np.ndarray]]
            ``(evaluation_grid_np, {cl: mean_size_per_grid_point, ...})`` if a sequence.
        """
        evaluation_grid_np = to_np_if_torch(evaluation_grid)
        if evaluation_grid_np.ndim == 1:
            evaluation_grid_np = evaluation_grid_np.reshape(-1, 1)
        n_grid = len(evaluation_grid_np)
        param_dim = evaluation_grid_np.shape[1]

        scalar_input = isinstance(confidence_level, float)
        cls: list = [confidence_level] if scalar_input else list(confidence_level)
        sizes: Dict[float, np.ndarray] = {cl: np.zeros(n_grid) for cl in cls}

        for i, theta_star in enumerate(evaluation_grid_np):
            if verbose and i % max(1, n_grid // 10) == 0:
                print(f'MC power: grid point {i}/{n_grid}', flush=True)
            theta_repeated = np.tile(theta_star.reshape(1, param_dim), (monte_carlo_size, 1))
            samples_mc = simulator(to_torch_if_np(theta_repeated))

            cs_batch = self.inference(
                x=samples_mc,
                evaluation_grid=evaluation_grid,
                confidence_level=confidence_level,
                calibration_method=calibration_method,
                calibration_model=self.calibration_model,
                verbose=False,
            )

            if scalar_input:
                sizes[cls[0]][i] = np.mean([cs.shape[0] / n_grid for cs in cs_batch])
            else:
                for j, cl in enumerate(cls):
                    sizes[cl][i] = np.mean([cs.shape[0] / n_grid for cs in cs_batch[j]])

            gc.collect()

        if scalar_input:
            return evaluation_grid_np, sizes[cls[0]]
        return evaluation_grid_np, sizes


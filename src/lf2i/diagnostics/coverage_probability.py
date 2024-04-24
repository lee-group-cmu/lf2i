from typing import Any, Dict, Optional, Tuple, Union, Sequence
import pathlib
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed

import rpy2.robjects as robj
import rpy2.robjects.numpy2ri
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier
import torch
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.utils.kde import KDEWrapper
from sbi.simulators.simutils import tqdm_joblib

from lf2i.test_statistics.waldo import Waldo
from lf2i.utils.calibration_diagnostics_inputs import (
    preprocess_indicators_lf2i, 
    preprocess_indicators_posterior, 
    preprocess_indicators_prediction, 
    preprocess_diagnostics
)
from lf2i.utils.other_methods import hpd_region, gaussian_prediction_sets


def estimate_coverage_proba(
    indicators: np.ndarray,
    parameters: np.ndarray,
    estimator: str,
    estimator_kwargs: Dict,
    param_dim: int,
    new_parameters: Optional[np.ndarray] = None,
    n_sigma: int = 2
) -> Tuple[Any, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    r"""Estimate conditional coverage probabilities by regressing `indicators`, which signal if the corresponding value in `parameters` 
    was included or not in the parameter region, against the `parameters` themselves. 

    Note that `indicators` can be computed from any parameter region (posterior credible sets, confidence sets, prediction sets, etc ...).

    Parameters
    ----------
    indicators : np.ndarray
        Array of zeros and ones to mark which `parameters` were included or not in the corresponding parameter regions. 
    parameters : np.ndarray
        Array of p-dimensional parameters.
    estimator : str
        Name of the probabilistic classifier to use to estimate coverage probabilities. 
    estimator_kwargs : Dict
        Settings for `estimator`.
    param_dim : int
        Dimensionality of the parameter.
    new_parameters : Optional[np.ndarray], optional
        Array of parameters over which to estimate/evaluate coverage probabilities.
        If not provided, both training and evaluation of the probabilistic classifier are done over `parameters`.
    n_sigma : int, optional
        Uncertainties around the estimated mean coverage proabilities are computed as :math:`\mu \pm se \cdot n\_sigma`.
        If using the `splines` estimator, the standard errors are based on the posterior distribution of the model coefficients. 
        By default 2.

    Returns
    -------
    Tuple[Any, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
        Fitted estimator, evaluated parameters, and estimated coverage probabilities -- mean, upper-n_sigma bound, lower-n_sigma bound. Bounds only available if `estimator="splines"`.

    Raises
    ------
    ValueError
        `Estimator` must be one of [`splines`, `cat-gb`].
    """
    indicators, parameters, new_parameters = preprocess_diagnostics(indicators, parameters, new_parameters, param_dim)
    if estimator  == 'splines':
        estimator = fit_r_estimator(
            estimator,
            indicators, 
            parameters,
            param_dim
        )
        mean_proba, upper_proba, lower_proba = predict_r_estimator(
            estimator, 
            parameters if new_parameters is None else new_parameters,
            param_dim,
            n_sigma
        )
    elif estimator == 'cat-gb':
        estimator = RandomizedSearchCV(
            estimator=CatBoostClassifier(
                loss_function='CrossEntropy',
                silent=True
            ),
            param_distributions={
                'iterations': [100, 300, 500, 700, 1000], 'depth': [1, 3, 5, 7, 10],
            },
            n_iter=25,
            n_jobs=-2,
            refit=True,
            cv=5
        )
        estimator.fit(X=parameters, y=indicators)
        best_params = estimator.best_params_

        estimator = CalibratedClassifierCV(
            estimator=CatBoostClassifier(
                loss_function='CrossEntropy',
                silent=True,
                **best_params
            ),
            method='isotonic',
            cv=5,
            n_jobs=-2
        )
        estimator.fit(X=parameters, y=indicators)
        mean_proba, upper_proba, lower_proba = estimator.predict(X=parameters if new_parameters is None else new_parameters), None, None
    else:
        # TODO: additional methods?
        raise ValueError(f"Estimators currently supported: [`splines`, `cat-gb`]; got {estimator}")
    out_parameters = parameters if new_parameters is None else new_parameters
    return estimator, out_parameters, mean_proba, upper_proba, lower_proba


def compute_indicators_lf2i(
    calibration_method: str,
    test_statistics: np.ndarray,
    parameters: np.ndarray,
    critical_values: Optional[np.ndarray],
    p_values: Optional[np.ndarray],
    alpha: Optional[float],
    acceptance_region: Optional[str],
    param_dim: int
) -> np.ndarray:
    """Construct an array of indicators which mark whether each value in `parameters` is included or not in the corresponding LF2I confidence region.
    
    This assumes that `parameters` is an array containing the “true” values, as simulated for the diagnostics branch.
    Instead of actually checking if the parameter is geometrically included in the confidence region or not, this allows to simply deem
    a value as included if the corresponding test does not reject it.

    Parameters
    ----------
    calibration_method : str
        Either `critical-values` or `p-values`.
    test_statistics : np.ndarray
        Array of test statistics. Each value must be computed for the test with corresponding (null) value of `parameters`, given a sample generate from it.
        Only used if `calibration_method = 'critical-values`. 
    parameters : np.ndarray
        True (simulated) parameter values. If a parameter is in the acceptance region of the corresponding test, then it is included in the confidence set. 
        Only used if `calibration_method = 'critical-values`.
    critical_values : np.ndarray, optional
        Array of critical values, each computed for the test with corresponding (null) value of `parameters`, against which to compare the test statistics.
        Only used if `calibration_method = 'critical-values`.
    p_values : np.ndarray, optional
        Array of p-values, each computed for the test with corresponding (null) value of `parameters`, against which to compare the provided level :math:`\alpha`.
        Only used if `calibration_method = 'p-values`.
    alpha: float, optional
        If `calibration_method = 'p-values`, used to decide whether the test rejects or not, otherwise ignored.
    acceptance_region : str, optional
        Whether the acceptance region for the corresponding test is defined to be on the right or on the left of the critical value. 
        Must be either `left` or `right`.  Only used if `calibration_method = 'critical-values`.
    param_dim : int
        Dimensionality of the parameter.
    Returns
    -------
    np.ndarray
        Array of zeros and ones that indicate whether the corresponding value in `parameters` is included or not in the confidence region.

    Raises
    ------
    ValueError
        `acceptance_region` must be either `left` or `right`.
    """
    # TODO: convert torch Tensors into numpy arrays
    test_statistics, critical_values, p_values, parameters = \
        preprocess_indicators_lf2i(test_statistics, critical_values, p_values, parameters, param_dim)
    
    indicators = np.zeros(shape=(parameters.shape[0], ))
    if calibration_method == 'critical-values':
        if acceptance_region == 'left':
            indicators[test_statistics <= critical_values] = 1
        elif acceptance_region == 'right':
            indicators[test_statistics >= critical_values] = 1
        else:
            raise ValueError(f"Acceptance region must be either `left` or `right`, got {acceptance_region}")
    else:
        indicators[p_values > alpha] = 1

    return indicators


def compute_indicators_posterior(
    posterior: Union[NeuralPosterior, KDEWrapper, Sequence[Union[NeuralPosterior, KDEWrapper]]],
    parameters: torch.Tensor,
    samples: torch.Tensor,
    parameter_grid: torch.Tensor,
    credible_level: float,
    param_dim: int,
    batch_size: int,
    num_level_sets: int = 10_000,
    tol: float = 0.01,
    norm_posterior_samples: Optional[int] = None,
    return_credible_regions: bool = False,
    verbose: bool = True,
    n_jobs: int = -2
) -> Union[np.ndarray, Tuple[np.ndarray, Sequence[np.ndarray]]]:
    """Construct an array of indicators which mark whether each value in `parameters` is included or not in the corresponding posterior credible region.

    Parameters
    ----------
    posterior : Union[NeuralPosterior, KDEWrapper, Sequence[Union[NeuralPosterior, KDEWrapper]]],
        Estimated posterior distribution. If `Sequence` of posteriors, we assume i-th posterior is estimated given i-th element of `samples`.
        Must have `log_prob()` method. 
    parameters : torch.Tensor,
        True (simulated) parameter values, for which inclusion in the corresponding credible region is checked.
    samples : torch.Tensor,
        Array of d-dimensional samples, each generated from the corresponding value in `parameters`.
    parameter_grid : torch.Tensor,
        Parameter space over which `posterior` is defined. This is used to construct the credible region. 
    credible_level : float
        Desired credible level for the HPD regions. Must be in (0, 1).
    param_dim : int
        Dimensionality of the parameter.
    batch_size : int
        Number of samples drawn from the same parameter value, for each batch in `samples`. 
        Each element of `samples` is of size (batch_size, data_dim).
    num_level_sets : int, optional
        Number of level sets to consider to construct the high-posterior-density region, by default 10_000.
        A high number of level sets ensures the actual credible level is as close as possible to the specified one.
    tol : float, optional
        Actual credible levels within `tol` of the specified `credible_level` will be considered acceptable, by default 0.01.
        NOTE: this is used as a stopping criterion, but if the closest actual credible level is not within `tol` of `credible_level`, a warning is raised but the HPD region is still used.
    norm_posterior_samples : int, optional
        Number of samples to use to estimate the leakage correction factor, by default None. More samples lead to better estimates of the normalization constant when returning a normalized posterior.
        If `None`, uses the un-normalized posterior (but note that the density is already being explicitly normalized over the `param_grid`).
    return_credible_regions: bool, optional
        Whether to return the credible regions computed along the way or not.
    verbose: bool, optional
        Whether to print progress bars or not, by default True.
    n_jobs : int, optional
        Number of workers to use when computing indicators over a sequence of inputs. By default -2, which uses all cores minus one.

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, Sequence[np.ndarray]]]
        Array of zeros and ones that indicates whether the corresponding value in `parameters` is included or not in the credible region.
        If `return_credible_regions`, then return a tuple whose second element is a sequence of credible regions (one for each parameter/sample).
    """
    parameters, samples, parameter_grid, posterior = \
        preprocess_indicators_posterior(parameters, samples, parameter_grid, param_dim, batch_size, posterior)
    
    def single_hpd_region(idx):
        _, credible_region = hpd_region(
            posterior=next(posterior),
            param_grid=torch.cat((parameter_grid, parameters[idx, :].reshape(1, param_dim))),
            x=samples[idx, :, :],
            credible_level=credible_level,
            num_level_sets=num_level_sets, tol=tol,
            norm_posterior_samples=norm_posterior_samples
        )
        # TODO: this is not safe. Better to return an array of bools and check if True
        indicator = 1 if parameters[idx, :] in credible_region else 0
        return credible_region, indicator

    with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Computing indicators for {len(it)} credible regions", total=len(it), disable=not verbose)) as _:
        out = list(zip(*Parallel(n_jobs=n_jobs)(delayed(single_hpd_region)(idx) for idx in it)))
    credible_regions, indicators = out[0], np.array(out[1])
    
    if return_credible_regions:
        return indicators, credible_regions
    else:
        return indicators


def compute_indicators_prediction(
    test_statistic: Waldo,
    parameters: np.ndarray,
    samples: np.ndarray,
    confidence_level: float,
    param_dim: int
) -> np.ndarray:
    """Construct an array of indicators which mark whether each value in `parameters` is included or not in the corresponding prediction set.
    The (central) prediction set is computed using a gaussian approximation.

    Parameters
    ----------
    test_statistic: Waldo
        An instance of the Waldo test statistic object, where Waldo.estimator and Waldo.cond_variance_estimator 
        have been trained and have a `predict(X=...)` method to estimate the conditional mean and conditional variance given `samples`.
    parameters : np.ndarray
        True (simulated) parameter values, for which inclusion in the corresponding prediction set is checked.
    confidence_level : float
        Confidence level of the credible regions to be constructed. Must be in (0, 1).
    param_dim : int
        Dimensionality of the parameter.

    Returns
    -------
    np.ndarray
        Array of zeros and ones that indicate whether the corresponding value in `parameters` is included or not in the prediction set.

    Raises
    ------
    NotImplementedError
        Only implemented for one-dimensional parameters.
    """
    parameters, samples = preprocess_indicators_prediction(parameters, samples, param_dim)
    if param_dim == 1:
        prediction_sets_bounds = gaussian_prediction_sets(
            conditional_mean_estimator=test_statistic.estimator,
            conditional_variance_estimator=test_statistic.cond_variance_estimator,
            samples=samples,
            confidence_level=confidence_level,
            param_dim=param_dim
        )
        indicators = np.zeros(shape=(parameters.shape[0], ))
        indicators[
            (
                (parameters >= prediction_sets_bounds[:, 0].reshape(-1, 1)) & 
                (parameters <= prediction_sets_bounds[:, 1].reshape(-1, 1))
            ).reshape(-1, )
        ] = 1 
    else:
        raise NotImplementedError

    return indicators


def fit_r_estimator(
    estimator: str,
    indicators: np.ndarray,
    parameters: np.ndarray,
    param_dim: int
) -> Any:
    """Estimate coverage probabilities across the whole parameter space using a pre-defined estimator available in R. 

    Parameters
    ----------
    estimator : str
        Name of the estimator to use.
    indicators : np.ndarray
        Array of zeros and ones that indicate whether the corresponding value in `parameters` is included or not in the parameter region.
    parameters : np.ndarray
        True (simulated) parameter values.
    param_dim : int
        Dimensionality of the parameter.

    Returns
    -------
    Any
        A model object returned by the corresponding R code.

    Raises
    ------
    NotImplementedError
        Estimator must be one of [`gam`, TBD]
    """
    file_path = pathlib.Path(__file__).parent.resolve() / 'estimators.r'
    robj.r(f"source('{file_path}')")
    robj.conversion.py2ri = robj.numpy2ri
    rpy2.robjects.numpy2ri.activate()

    indicators = robj.r.matrix(indicators)
    parameters = robj.r.matrix(parameters.reshape((parameters.size)), nrow=parameters.shape[0], ncol=parameters.shape[1], byrow=True)

    if estimator == 'splines':
        try:
            output_dict = robj.globalenv['fit_joint_splines'](indicators, parameters, param_dim)
        except Exception as e:
            estimator = 'gam_splines'
            # NOTE: memoryerror can occur if joint splines tensor is too big. Default behaviour is to use additive splines (one for each input) for now.
            warnings.warn(f'Training joint tensor product splines basis raised {str(e)}. Reverting to additive splines via GAMs.')
            output_dict = robj.globalenv['fit_additive_splines'](indicators, parameters, param_dim)
    else: 
        # TODO: additional methods?
        raise NotImplementedError(f"Estimator must be one of [`splines`], got {estimator}")

    output_dict = dict(zip(output_dict.names, list(output_dict)))
    return output_dict[estimator]


def predict_r_estimator(
    fitted_estimator: Any,  # r object
    parameters: np.ndarray,
    param_dim: int,
    n_sigma: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the trained R estimator and estimate the coverage probabilities given `parameters`.

    Parameters
    ----------
    fitted_estimator : Any
        Trained estimator, as returned by `fit_r_estimator`.
    param_dim : int
        Dimensionality of the parameter.
    n_sigma : int
        Uncertainties around the estimated mean coverage proabilities are computed as mean +- se * n_sigma, by default 2.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Estimated conditional coverage probabilities -- mean, upper-n_sigma bound. lower-n_sigma bound
    """
    file_path = pathlib.Path(__file__).parent.resolve() / 'estimators.r'
    robj.r(f"source('{file_path}')")
    robj.conversion.py2ri = robj.numpy2ri
    rpy2.robjects.numpy2ri.activate()

    parameters = robj.r.matrix(parameters.reshape((parameters.size)), nrow=parameters.shape[0], ncol=parameters.shape[1], byrow=True)
    # TODO: this assumes a gam-like model
    output_dict = robj.globalenv['predict_gam'](fitted_estimator, parameters, param_dim)
    output_dict = dict(zip(output_dict.names, list(output_dict)))

    mean_proba = np.array(output_dict["predictions"])
    upper_proba = np.maximum(0, np.minimum(1, mean_proba + np.array(output_dict["se"]) * n_sigma))
    lower_proba = np.maximum(0, np.minimum(1, mean_proba - np.array(output_dict["se"]) * n_sigma))
    return mean_proba, upper_proba, lower_proba
